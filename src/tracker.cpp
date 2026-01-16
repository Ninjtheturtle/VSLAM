#include "slam/tracker.hpp"
#include "slam/local_ba.hpp"
#include "slam/map_point.hpp"
#include "cuda/hamming_matcher.cuh"

#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/SVD>
#include <iostream>
#include <atomic>
#include <unordered_set>
#include <numeric>

namespace slam {

static std::atomic<long> g_frame_id{0};  // global atomic ID counters
static std::atomic<long> g_point_id{0};

// pose conversion for motion-only BA
static void isometry_to_pose_tracker(const Eigen::Isometry3d& T, double* p) {
    Eigen::Quaterniond q(T.rotation());
    q.normalize();
    p[0] = q.x(); p[1] = q.y(); p[2] = q.z(); p[3] = q.w();
    p[4] = T.translation().x();
    p[5] = T.translation().y();
    p[6] = T.translation().z();
}

Tracker::Ptr Tracker::create(const Camera& cam, Map::Ptr map, const Config& cfg)
{
    auto t = std::shared_ptr<Tracker>(new Tracker());
    t->cam_ = cam;
    t->map_ = map;
    t->cfg_ = cfg;
    t->orb_ = cv::ORB::create(
        cfg.orb_features,
        cfg.orb_scale_factor,
        cfg.orb_levels,
        cfg.orb_edge_threshold
    );
    return t;
}

// ORB detect+describe on left (and right, if stereo). called only at KFs ,
// per-frame tracking is KLT on the cached pyramid and touches no descriptors
void Tracker::extract_kf_features(Frame::Ptr frame)
{
    orb_->detectAndCompute(frame->image_gray, cv::noArray(),
                           frame->keypoints, frame->descriptors);
    if (!frame->keypoints.empty()) {
        std::vector<cv::Point2f> corners(frame->keypoints.size());
        for (size_t k = 0; k < frame->keypoints.size(); ++k)
            corners[k] = frame->keypoints[k].pt;
        cv::cornerSubPix(frame->image_gray, corners,
                         cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20, 0.01));
        for (size_t k = 0; k < frame->keypoints.size(); ++k)
            frame->keypoints[k].pt = corners[k];
    }
    frame->map_points.assign(frame->keypoints.size(), nullptr);

    if (!frame->image_right.empty()) {
        orb_->detectAndCompute(frame->image_right, cv::noArray(),
                               frame->keypoints_right, frame->descriptors_right);
        if (!frame->keypoints_right.empty()) {
            std::vector<cv::Point2f> corners_r(frame->keypoints_right.size());
            for (size_t k = 0; k < frame->keypoints_right.size(); ++k)
                corners_r[k] = frame->keypoints_right[k].pt;
            cv::cornerSubPix(frame->image_right, corners_r,
                             cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20, 0.01));
            for (size_t k = 0; k < frame->keypoints_right.size(); ++k)
                frame->keypoints_right[k].pt = corners_r[k];
        }
        frame->uR.assign(frame->keypoints.size(), -1.0f);
        match_stereo(frame);
    }
}

bool Tracker::track(Frame::Ptr frame)
{
    // LOST: reset the map and re-initialize from the next frame
    if (state_ == TrackingState::LOST) {
        std::cerr << "[Tracker] LOST — resetting map + re-initializing\n";
        frame->T_cw = last_frame_->T_cw;
        map_->reset();
        last_keyframe_       = nullptr;
        last_kf_pnp_tracked_ = 0;
        frames_since_kf_     = 0;
        velocity_valid_      = false;
        last_frame_          = frame;
        state_               = TrackingState::NOT_INITIALIZED;
        return false;
    }

    if (state_ == TrackingState::NOT_INITIALIZED)
        return initialize(frame);

    // OK: KLT flow -> motion-only BA -> maybe KF
    bool ok = track_with_klt(frame);
    if (ok) ok = track_local_map(frame);
    if (!ok) state_ = TrackingState::LOST;
    last_frame_ = frame;
    return ok;
}

bool Tracker::initialize(Frame::Ptr frame)
{
    // init must extract features so we have stereo correspondences for triangulation
    extract_kf_features(frame);

    if (!cam_.is_stereo() || frame->uR.empty()) {
        std::cerr << "[Tracker] Init: stereo-only mode but no stereo data, skipping\n";
        last_frame_ = frame;
        return false;
    }
    if (last_frame_) frame->T_cw = last_frame_->T_cw;

    int n_pts = triangulate_stereo(frame);
    if (n_pts < 50) {
        std::cerr << "[Tracker] Stereo init: too few points (" << n_pts << "), retrying\n";
        last_frame_ = frame;
        return false;
    }
    insert_keyframe(frame);
    state_      = TrackingState::OK;
    last_frame_ = frame;
    std::cout << "[Tracker] Stereo initialized: " << n_pts << " metric map points\n";
    return true;
}

// bucket flow tracks into a spatial grid and keep at most one per cell, biased
// toward fresh tracks (lowest observed_times). Then truncate to max_flow_tracks
void Tracker::cap_flow_tracks(Frame::Ptr frame) const
{
    if ((int)frame->flow_pts.size() <= cfg_.max_flow_tracks) return;
    if (cam_.width <= 0 || cam_.height <= 0) return;

    const int gc = cfg_.flow_grid_cols;
    const int gr = cfg_.flow_grid_rows;
    std::vector<int> best_in_cell(gc * gr, -1);

    auto cell_of = [&](const cv::Point2f& p) {
        int cx = std::min(gc - 1, std::max(0, int(p.x * gc / cam_.width)));
        int cy = std::min(gr - 1, std::max(0, int(p.y * gr / cam_.height)));
        return cy * gc + cx;
    };

    // prefer the most-observed (longest-lived, best-validated) track per cell
    // long tracks survived multiple BA cycles so lower reproj error,
    // less drift contribution. Fresh tracks are noisy and bias BA badly
    for (int i = 0; i < (int)frame->flow_pts.size(); ++i) {
        int c = cell_of(frame->flow_pts[i]);
        int b = best_in_cell[c];
        if (b < 0 ||
            frame->flow_mps[i]->observed_times > frame->flow_mps[b]->observed_times) {
            best_in_cell[c] = i;
        }
    }

    std::vector<int> keep;
    keep.reserve(gc * gr);
    for (int c : best_in_cell) if (c >= 0) keep.push_back(c);

    if ((int)keep.size() < cfg_.max_flow_tracks) {
        std::vector<bool> taken(frame->flow_pts.size(), false);
        for (int i : keep) taken[i] = true;
        std::vector<int> rest;
        for (int i = 0; i < (int)frame->flow_pts.size(); ++i)
            if (!taken[i]) rest.push_back(i);
        std::sort(rest.begin(), rest.end(), [&](int a, int b){
            return frame->flow_mps[a]->observed_times > frame->flow_mps[b]->observed_times;
        });
        for (int i : rest) {
            if ((int)keep.size() >= cfg_.max_flow_tracks) break;
            keep.push_back(i);
        }
    } else if ((int)keep.size() > cfg_.max_flow_tracks) {
        std::sort(keep.begin(), keep.end(), [&](int a, int b){
            return frame->flow_mps[a]->observed_times > frame->flow_mps[b]->observed_times;
        });
        keep.resize(cfg_.max_flow_tracks);
    }

    std::vector<cv::Point2f>   new_pts;
    std::vector<MapPoint::Ptr> new_mps;
    new_pts.reserve(keep.size());
    new_mps.reserve(keep.size());
    for (int i : keep) {
        new_pts.push_back(frame->flow_pts[i]);
        new_mps.push_back(frame->flow_mps[i]);
    }
    frame->flow_pts = std::move(new_pts);
    frame->flow_mps = std::move(new_mps);
}

// KLT tracking
//
// replaces descriptor matching entirely on non-KF frames. Tracks the previous frame's
// (flow_pts, flow_mps) into the current frame using image pyramids cached on each Frame
// fwd-backward consistency rejects KLT outliers. The surviving 2D-3D correspondences
// feed motion-only Ceres BA directly , no constant-velocity prediction, no search radius

bool Tracker::track_with_klt(Frame::Ptr frame)
{
    // constant-velocity prediction as MoBA initial guess
    // Without this, the optimizer starts ~1m off at KITTI speeds (10Hz, ~10m/s)
    if (velocity_valid_) {
        frame->T_cw = velocity_ * last_frame_->T_cw;
    } else {
        frame->T_cw = last_frame_->T_cw;
    }

    if (!last_frame_ || last_frame_->flow_pts.empty() ||
        last_frame_->klt_pyramid.empty() || frame->klt_pyramid.empty()) {
        std::cerr << "[KLT] No previous flow tracks to propagate\n";
        return false;
    }

    const std::vector<cv::Point2f>& prev_pts = last_frame_->flow_pts;
    std::vector<cv::Point2f>        cur_pts;
    std::vector<uchar>              status;
    std::vector<float>              err;
    cv::Size win(cfg_.klt_win, cfg_.klt_win);

    cv::calcOpticalFlowPyrLK(last_frame_->klt_pyramid, frame->klt_pyramid,
                             prev_pts, cur_pts, status, err, win, cfg_.klt_max_level,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

    // fwd-backward consistency: reproject current pts back into prev frame
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar>       status_b;
    std::vector<float>       err_b;
    cv::calcOpticalFlowPyrLK(frame->klt_pyramid, last_frame_->klt_pyramid,
                             cur_pts, back_pts, status_b, err_b, win, cfg_.klt_max_level,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

    const int W = frame->image_gray.cols;
    const int H = frame->image_gray.rows;

    // build pruned flow_pts/flow_mps + parallel pts3d/pts2d for BA
    std::vector<cv::Point3f>               pts3d;
    std::vector<cv::Point2f>               pts2d;
    std::vector<MapPoint::Ptr>             surv_mps;
    std::vector<cv::Point2f>               surv_pts;
    pts3d.reserve(prev_pts.size());
    pts2d.reserve(prev_pts.size());
    surv_mps.reserve(prev_pts.size());
    surv_pts.reserve(prev_pts.size());

    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status[i] || !status_b[i]) continue;
        float dx = back_pts[i].x - prev_pts[i].x;
        float dy = back_pts[i].y - prev_pts[i].y;
        if (dx*dx + dy*dy > cfg_.klt_fb_max_err * cfg_.klt_fb_max_err) continue;

        const cv::Point2f& p = cur_pts[i];
        if (p.x < 1 || p.y < 1 || p.x >= W - 1 || p.y >= H - 1) continue;

        auto& mp = last_frame_->flow_mps[i];
        if (!mp || mp->is_bad) continue;

        const Eigen::Vector3d& X = mp->position;
        pts3d.push_back({(float)X.x(), (float)X.y(), (float)X.z()});
        pts2d.push_back(p);
        surv_mps.push_back(mp);
        surv_pts.push_back(p);
    }

    fprintf(stderr, "[KLT] tracked %d / %d (FB-filtered)\n",
            (int)pts2d.size(), (int)prev_pts.size());

    if ((int)pts2d.size() < cfg_.pnp_min_inliers) {
        std::cerr << "[KLT] too few tracks (" << pts2d.size() << ")\n";
        return false;
    }

    // Motion-only Ceres BA on KLT correspondences (mono only)
    // stereo KLT per frame was removed , it's redundant because the 3D map points
    // are already at metric positions from stereo triangulation. With fixed 3D points,
    // mono reprojection fully constrains the 6-DOF pose including translation scale
    Eigen::Isometry3d T_cw_candidate;
    {
        double pose[7];
        isometry_to_pose_tracker(frame->T_cw, pose);

        ceres::Problem moba_problem;
        ceres::LossFunction* moba_loss = new ceres::HuberLoss(cfg_.moba_huber);
        const double info_uv = 1.0;

        std::vector<std::array<double, 3>> pt_blocks;
        pt_blocks.reserve(pts3d.size());

        for (size_t i = 0; i < pts3d.size(); ++i) {
            pt_blocks.push_back({(double)pts3d[i].x, (double)pts3d[i].y, (double)pts3d[i].z});
            double* pt = pt_blocks.back().data();

            Eigen::Vector3d Pw(pt[0], pt[1], pt[2]);
            double Zc = (frame->T_cw.rotation() * Pw + frame->T_cw.translation()).z();
            if (Zc <= 0.01) { pt_blocks.pop_back(); continue; }

            moba_problem.AddResidualBlock(
                MonoReprojCost::Create(pts2d[i].x, pts2d[i].y,
                    cam_.fx, cam_.fy, cam_.cx, cam_.cy, info_uv),
                moba_loss, pose, pt);
            moba_problem.SetParameterBlockConstant(pt);
        }

        moba_problem.AddParameterBlock(pose, 7,
            new ceres::ProductManifold(new ceres::EigenQuaternionManifold,
                                       new ceres::EuclideanManifold<3>));

        ceres::Solver::Options moba_opts;
        moba_opts.linear_solver_type = ceres::DENSE_QR;
        moba_opts.max_num_iterations = cfg_.moba_iterations;
        moba_opts.num_threads = 1;
        ceres::Solver::Summary moba_summary;
        ceres::Solve(moba_opts, &moba_problem, &moba_summary);

        Eigen::Quaterniond q_opt(pose[3], pose[0], pose[1], pose[2]);
        q_opt.normalize();
        T_cw_candidate = Eigen::Isometry3d::Identity();
        T_cw_candidate.linear()      = q_opt.toRotationMatrix();
        T_cw_candidate.translation() << pose[4], pose[5], pose[6];
    }

    // sanity-check delta motion
    {
        Eigen::Isometry3d delta = T_cw_candidate * last_frame_->T_cw.inverse();
        double da = Eigen::AngleAxisd(delta.rotation()).angle();
        double dt = delta.translation().norm();
        if (da > 0.3) { fprintf(stderr, "[KLT] reject delta-rot %.1f deg\n", da*57.2958); return false; }
        if (dt > 3.0) { fprintf(stderr, "[KLT] reject delta-trans %.1f m\n", dt); return false; }
    }

    // update constant-velocity model from the optimized pose
    velocity_ = T_cw_candidate * last_frame_->T_cw.inverse();
    velocity_valid_ = true;

    frame->T_cw = T_cw_candidate;

    // reprojection cull at the optimized pose, then commit surviving tracks
    frame->flow_pts.clear();
    frame->flow_mps.clear();
    frame->flow_pts.reserve(surv_pts.size());
    frame->flow_mps.reserve(surv_mps.size());

    const double cull_thresh2 = 9.0;  // 3px reprojection
    int n_inliers = 0;
    for (size_t i = 0; i < surv_mps.size(); ++i) {
        Eigen::Vector3d Xc = frame->T_cw * surv_mps[i]->position;
        if (Xc.z() <= 0.0) continue;
        double u = cam_.fx * Xc.x() / Xc.z() + cam_.cx;
        double v = cam_.fy * Xc.y() / Xc.z() + cam_.cy;
        double du = u - surv_pts[i].x;
        double dv = v - surv_pts[i].y;
        if (du*du + dv*dv > cull_thresh2) continue;

        frame->flow_pts.push_back(surv_pts[i]);
        frame->flow_mps.push_back(surv_mps[i]);
        surv_mps[i]->observed_times++;
        ++n_inliers;
    }

    if (n_inliers < cfg_.pnp_min_inliers) {
        std::cerr << "[KLT] post-BA inliers too few (" << n_inliers << ")\n";
        return false;
    }

    cap_flow_tracks(frame);
    fprintf(stderr, "[KLT] BA inliers: %d (capped to %d)\n",
            n_inliers, (int)frame->flow_pts.size());
    return true;
}


// check if we should insert a keyframe after successful tracking

bool Tracker::track_local_map(Frame::Ptr frame)
{
    ++frames_since_kf_;
    if (need_new_keyframe(frame)) {
        insert_keyframe(frame);
    }
    return (int)frame->flow_mps.size() >= cfg_.pnp_min_inliers;
}

// KF decision , multi-criteria trigger

bool Tracker::need_new_keyframe(Frame::Ptr frame) const
{
    if (!last_keyframe_) return true;

    // (1) starving , too few surviving KLT tracks
    if ((int)frame->flow_mps.size() < cfg_.min_tracked_points) return true;

    // (2) AABB collapse , surviving tracks no longer span the image margins
    //     Trigger when the bounding box of all tracked pixels covers
    //     less than kf_min_aabb_frac of the image area (default 30%)
    if (!frame->flow_pts.empty() && cam_.width > 0 && cam_.height > 0) {
        float xmin =  std::numeric_limits<float>::max();
        float ymin =  std::numeric_limits<float>::max();
        float xmax = -std::numeric_limits<float>::max();
        float ymax = -std::numeric_limits<float>::max();
        for (const auto& p : frame->flow_pts) {
            if (p.x < xmin) xmin = p.x;
            if (p.y < ymin) ymin = p.y;
            if (p.x > xmax) xmax = p.x;
            if (p.y > ymax) ymax = p.y;
        }
        float aabb_area = std::max(0.f, xmax - xmin) * std::max(0.f, ymax - ymin);
        float img_area  = float(cam_.width) * float(cam_.height);
        if (aabb_area < cfg_.kf_min_aabb_frac * img_area) return true;
    }

    Eigen::Isometry3d delta = frame->T_cw * last_keyframe_->T_cw.inverse();

    if (delta.translation().norm() > cfg_.kf_min_translation) return true;
    if (Eigen::AngleAxisd(delta.rotation()).angle() > cfg_.kf_min_rotation) return true;
    if (frames_since_kf_ >= cfg_.kf_max_frames) return true;

    return false;
}

double Tracker::compute_median_parallax(Frame::Ptr frame, Frame::Ptr ref_kf) const
{
    if (!ref_kf) return 0.0;
    const Eigen::Vector3d C_ref = ref_kf->camera_center();
    const Eigen::Vector3d C_cur = frame->camera_center();
    std::vector<double> angles;
    for (int i = 0; i < (int)frame->map_points.size(); ++i) {
        auto& mp = frame->map_points[i];
        if (!mp || mp->is_bad) continue;
        Eigen::Vector3d v1 = (mp->position - C_ref).normalized();
        Eigen::Vector3d v2 = (mp->position - C_cur).normalized();
        double cos_a = std::max(-1.0, std::min(1.0, v1.dot(v2)));
        angles.push_back(std::acos(std::abs(cos_a)));
    }
    if (angles.empty()) return 0.0;
    std::sort(angles.begin(), angles.end());
    return angles[angles.size() / 2];
}

void Tracker::log_anms_grid_stats(const std::vector<cv::KeyPoint>& kps,
                                  int img_w, int img_h,
                                  int grid_cols, int grid_rows) const
{
    std::vector<int> counts(grid_cols * grid_rows, 0);
    for (const auto& kp : kps) {
        int col = std::min((int)(kp.pt.x / img_w * grid_cols), grid_cols - 1);
        int row = std::min((int)(kp.pt.y / img_h * grid_rows), grid_rows - 1);
        ++counts[row * grid_cols + col];
    }
    float mean = (float)kps.size() / counts.size();
    float var = 0.f;
    for (int c : counts) var += (c - mean) * (c - mean);
    float cv = std::sqrt(var / counts.size()) / (mean + 1e-6f);
    fprintf(stderr, "[ANMS] %d pts | grid %dx%d | mean/cell=%.1f | CoV=%.2f\n",
            (int)kps.size(), grid_cols, grid_rows, mean, cv);
    for (int r = 0; r < grid_rows; ++r) {
        int row_total = 0;
        for (int c = 0; c < grid_cols; ++c) row_total += counts[r * grid_cols + c];
        fprintf(stderr, "  row%d: %d pts (%.0f%%)\n",
                r, row_total, 100.f * row_total / ((int)kps.size() + 1));
    }
}

void Tracker::insert_keyframe(Frame::Ptr frame)
{
    frames_since_kf_ = 0;

    // KEYFRAME-ONLY descriptor extraction. Per-frame tracking uses KLT only;
    // ORB is run here so this KF carries descriptors for triangulation,
    // PGO, loop closure, and BA
    if (frame->keypoints.empty()) {
        extract_kf_features(frame);
    }

    // graft the live KLT-tracked correspondences onto the keyframe's keypoint list
    // as extra observations so local BA optimizes against them too. These appended
    // entries have no descriptor row, local BA only consults keypoints[i].pt + uR[i]
    int n_kf_kps_pre = (int)frame->keypoints.size();
    for (size_t i = 0; i < frame->flow_mps.size(); ++i) {
        cv::KeyPoint kp;
        kp.pt = frame->flow_pts[i];
        kp.response = 1.0f;
        frame->keypoints.push_back(kp);
        frame->map_points.push_back(frame->flow_mps[i]);
        if (!frame->uR.empty()) frame->uR.push_back(-1.0f);
    }

    // stereo match grafted KLT tracks: single forward pass using pre-built pyramids
    // epipolar + disparity filter is sufficient for rectified stereo (no FB check needed)
    int n_stereo_grafted = 0;
    if (!frame->image_right.empty() && !frame->flow_pts.empty() && !frame->uR.empty()) {
        // build right-image pyramid once, cache on frame
        if (frame->klt_pyramid_right.empty()) {
            cv::buildOpticalFlowPyramid(frame->image_right, frame->klt_pyramid_right,
                                        cv::Size(cfg_.klt_win, cfg_.klt_win), cfg_.klt_max_level,
                                        /*withDerivatives=*/true);
        }

        std::vector<cv::Point2f> left_graft_pts(frame->flow_pts.begin(), frame->flow_pts.end());
        std::vector<cv::Point2f> right_graft_pts;
        std::vector<uchar>       st_stereo;
        std::vector<float>       err_stereo;

        cv::calcOpticalFlowPyrLK(frame->klt_pyramid, frame->klt_pyramid_right,
                                 left_graft_pts, right_graft_pts,
                                 st_stereo, err_stereo,
                                 cv::Size(cfg_.klt_win, cfg_.klt_win), cfg_.klt_max_level,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

        for (size_t i = 0; i < left_graft_pts.size(); ++i) {
            if (!st_stereo[i]) continue;
            float dy = std::abs(right_graft_pts[i].y - left_graft_pts[i].y);
            if (dy > cfg_.stereo_epi_tol) continue;
            float disp = left_graft_pts[i].x - right_graft_pts[i].x;
            if (disp < cfg_.stereo_d_min || disp > cfg_.stereo_d_max) continue;

            int kf_idx = n_kf_kps_pre + (int)i;
            if (kf_idx < (int)frame->uR.size()) {
                frame->uR[kf_idx] = right_graft_pts[i].x;
                ++n_stereo_grafted;
            }
        }
    }
    fprintf(stderr, "[KF] grafted %d KLT tracks onto %d extracted kps (%d stereo-matched)\n",
            (int)frame->flow_mps.size(), n_kf_kps_pre, n_stereo_grafted);

    if (cam_.width > 0 && !frame->keypoints.empty())
        log_anms_grid_stats(frame->keypoints, cam_.width, cam_.height, 6, 4);

    last_kf_pnp_tracked_ = frame->num_tracked();

    frame->id = g_frame_id++;
    frame->is_keyframe = true;

    // triangulate against last 3 KFs , oldest first for widest baseline
    for (auto& tri_kf : map_->local_window(3)) {
        if (tri_kf->id == frame->id) continue;
        auto kf_matches = match_descriptors(tri_kf->descriptors,
                                            frame->descriptors, /*ratio=*/true);
        std::vector<cv::DMatch> new_matches;
        for (auto& m : kf_matches) {
            if (m.trainIdx < (int)frame->map_points.size() &&
                !frame->map_points[m.trainIdx]) {
                new_matches.push_back(m);
            }
        }
        if (!new_matches.empty()) {
            int n_new = triangulate_and_add(tri_kf, frame, new_matches);
            if (n_new > 0)
                std::cout << "[Tracker] KF " << frame->id
                          << ": triangulated " << n_new
                          << " pts vs KF " << tri_kf->id << "\n";
        }
    }

    // stereo enrichment: fill unmapped kps w/ metric-depth points
    if (cam_.is_stereo() && !frame->uR.empty()) {
        int n_stereo = triangulate_stereo(frame);
        if (n_stereo > 0)
            std::cout << "[Tracker] KF " << frame->id
                      << ": stereo added " << n_stereo << " metric pts\n";
    }

    map_->insert_keyframe(frame);
    last_keyframe_ = frame;

// re-seed KLT flow tracks for the NEXT frame from every kp that now carries
    // a valid map point (extracted + stereo-triangulated + grafted-from-prev-KLT)
    frame->flow_pts.clear();
    frame->flow_mps.clear();
    frame->flow_pts.reserve(frame->map_points.size());
    frame->flow_mps.reserve(frame->map_points.size());
    for (size_t i = 0; i < frame->map_points.size(); ++i) {
        auto& mp = frame->map_points[i];
        if (!mp || mp->is_bad) continue;
        if (i >= frame->keypoints.size()) break;
        frame->flow_pts.push_back(frame->keypoints[i].pt);
        frame->flow_mps.push_back(mp);
    }
    cap_flow_tracks(frame);
    fprintf(stderr, "[KF] reseeded %d KLT tracks (capped)\n", (int)frame->flow_pts.size());
}

// GPU hamming matching via CUDA kernel

std::vector<cv::DMatch> Tracker::match_descriptors(
    const cv::Mat& query_desc,
    const cv::Mat& train_desc,
    bool use_ratio)
{
    int N_q = query_desc.rows;
    int N_t = train_desc.rows;

    if (N_q == 0 || N_t == 0) return {};

    // CUDA kernel needs contiguous CV_8U data
    cv::Mat q = query_desc.isContinuous() ? query_desc : query_desc.clone();
    cv::Mat t = train_desc.isContinuous()  ? train_desc  : train_desc.clone();

    std::vector<int> best_idx(N_q, -1);
    std::vector<int> best_dist(N_q, kMaxHamming);

    if (use_ratio) {
        cuda_match_hamming_ratio(
            q.data, t.data, N_q, N_t, cfg_.lowe_ratio,
            best_idx.data(), best_dist.data());
    } else {
        cuda_match_hamming(
            q.data, t.data, N_q, N_t,
            best_idx.data(), best_dist.data());
    }

    std::vector<cv::DMatch> matches;
    matches.reserve(N_q);
    for (int i = 0; i < N_q; ++i) {
        if (best_idx[i] >= 0 && best_dist[i] <= cfg_.hamming_threshold) {
            matches.push_back(cv::DMatch(i, best_idx[i], (float)best_dist[i]));
        }
    }
    return matches;
}

// triangulate new map points from matched keypoints between two KFs

int Tracker::triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                                  const std::vector<cv::DMatch>& matches)
{
    // build 3x4 projection matrices
    auto make_proj = [&](const Eigen::Isometry3d& T_cw) -> cv::Mat {
        cv::Mat P(3, 4, CV_64F);
        Eigen::Matrix<double, 3, 4> Rt;
        Rt.block<3,3>(0,0) = T_cw.rotation();
        Rt.block<3,1>(0,3) = T_cw.translation();
        Eigen::Matrix<double, 3, 4> KRt = cam_.K() * Rt;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                P.at<double>(r, c) = KRt(r, c);
        return P;
    };

    cv::Mat P0 = make_proj(ref->T_cw);
    cv::Mat P1 = make_proj(cur->T_cw);

    std::vector<cv::Point2f> pts0, pts1;
    std::vector<int>         ref_kp_idxs, cur_kp_idxs;
    for (auto& m : matches) {
        pts0.push_back(ref->keypoints[m.queryIdx].pt);
        pts1.push_back(cur->keypoints[m.trainIdx].pt);
        ref_kp_idxs.push_back(m.queryIdx);
        cur_kp_idxs.push_back(m.trainIdx);
    }

    cv::Mat pts4d;
    cv::triangulatePoints(P0, P1, pts0, pts1, pts4d);  // 4xN homogeneous

    int n_added = 0;
    for (int i = 0; i < pts4d.cols; ++i) {
        float w = pts4d.at<float>(3, i);  // triangulatePoints outputs CV_32F
        if (std::abs(w) < 1e-6f) continue;

        Eigen::Vector3d Xw(pts4d.at<float>(0, i) / w,
                           pts4d.at<float>(1, i) / w,
                           pts4d.at<float>(2, i) / w);

        // depth check in both cameras
        Eigen::Vector3d Xc0 = ref->T_cw * Xw;
        Eigen::Vector3d Xc1 = cur->T_cw * Xw;
        if (Xc0.z() < 0.05 || Xc1.z() < 0.05) continue;
        if (Xc0.z() > 100.0 || Xc1.z() > 100.0) continue;

        // reject near-degenerate triangulations , cos_pa > 0.9998 ≈ <1.1° parallax
        {
            Eigen::Vector3d O0 = ref->camera_center();
            Eigen::Vector3d O1 = cur->camera_center();
            double cos_pa = std::abs((Xw - O0).normalized().dot((Xw - O1).normalized()));
            if (cos_pa > 0.9998) continue;
        }

        auto mp = MapPoint::create(Xw, g_point_id++);
        mp->add_observation(ref->id, ref_kp_idxs[i]);
        mp->add_observation(cur->id, cur_kp_idxs[i]);

        ref->map_points[ref_kp_idxs[i]] = mp;
        cur->map_points[cur_kp_idxs[i]] = mp;

        map_->insert_map_point(mp);
        map_->update_covisibility(ref->id, mp);  // links ref & cur in covis graph
        map_->update_covisibility(cur->id, mp);
        ++n_added;
    }
    return n_added;
}

// relocalization , match against all KFs with stricter inlier threshold

bool Tracker::try_relocalize(Frame::Ptr frame)
{
    // build pool from ALL KFs , slow but fine since we're already LOST
    cv::Mat                    pool_desc;
    std::vector<MapPoint::Ptr> pool_mps;
    {
        std::unordered_set<long> seen_ids;
        for (auto& kf : map_->all_keyframes()) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (!seen_ids.insert(mp->id).second) continue;
                if (i >= kf->descriptors.rows) continue;
                pool_desc.push_back(kf->descriptors.row(i));
                pool_mps.push_back(mp);
            }
        }
    }
    if (pool_desc.rows < cfg_.pnp_min_inliers) return false;

    std::cout << "[Reloc] Matching against " << pool_desc.rows << " global map pts\n";

    auto raw_matches = match_descriptors(pool_desc, frame->descriptors, true);

    std::vector<cv::Point3f>   pts3d;
    std::vector<cv::Point2f>   pts2d;
    std::vector<int>           match_idxs;
    std::vector<MapPoint::Ptr> match_mps;
    std::unordered_set<int>    used_kp;

    for (auto& m : raw_matches) {
        if (!used_kp.insert(m.trainIdx).second) continue;
        auto& mp = pool_mps[m.queryIdx];
        auto& p  = mp->position;
        pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
        pts2d.push_back(frame->keypoints[m.trainIdx].pt);
        match_idxs.push_back(m.trainIdx);
        match_mps.push_back(mp);
    }
    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        std::cerr << "[Reloc] Too few correspondences (" << pts3d.size() << ")\n";
        return false;
    }
    std::cout << "[Reloc] " << pts3d.size() << " 3D-2D correspondences\n";

    // PnP ransac , no initial guess (velocity not trustworthy when LOST)
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat inlier_mask;

    int reloc_pnp = ((int)pts3d.size() < 30) ? cv::SOLVEPNP_EPNP : cv::SOLVEPNP_SQPNP;
    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, cam_.K_cv(), cam_.dist_cv(),
        rvec, tvec, /*useExtrinsicGuess=*/false,
        cfg_.pnp_iterations, cfg_.pnp_reprojection, 0.99,
        inlier_mask, reloc_pnp);

    if (!ok) { std::cerr << "[Reloc] solvePnPRansac failed\n"; return false; }

    std::vector<bool> is_inlier(pts3d.size(), false);
    int n_reloc_inliers = 0;
    if (inlier_mask.type() == CV_8U) {
        for (int k = 0; k < inlier_mask.rows && k < (int)pts3d.size(); ++k)
            if (inlier_mask.at<uint8_t>(k)) { is_inlier[k] = true; ++n_reloc_inliers; }
    } else {
        for (int k = 0; k < inlier_mask.rows; ++k) {
            int idx = inlier_mask.at<int>(k);
            if (idx >= 0 && idx < (int)pts3d.size()) { is_inlier[idx] = true; ++n_reloc_inliers; }
        }
    }

    const int reloc_min = cfg_.pnp_min_inliers * 2;  // stricter than normal tracking , was 30
    if (n_reloc_inliers < reloc_min) {
        std::cerr << "[Reloc] Inliers too few (" << n_reloc_inliers
                  << " < " << reloc_min << ")\n";
        return false;
    }

    // LM refinement on inliers
    std::vector<cv::Point3f> in3d;
    std::vector<cv::Point2f> in2d;
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i]) { in3d.push_back(pts3d[i]); in2d.push_back(pts2d[i]); }
    cv::solvePnP(in3d, in2d, cam_.K_cv(), cam_.dist_cv(),
                 rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            frame->T_cw.linear()(r, c) = R_cv.at<double>(r, c);
    frame->T_cw.translation() << tvec.at<double>(0),
                                  tvec.at<double>(1),
                                  tvec.at<double>(2);

    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i])
            frame->map_points[match_idxs[i]] = match_mps[i];

    std::cout << "[Reloc] SUCCESS — " << inlier_mask.rows << " inliers\n";
    return true;
}

// stereo epipolar matching

void Tracker::match_stereo(Frame::Ptr frame)
{
    if (frame->descriptors.empty() || frame->descriptors_right.empty()) return;
    int N_q = frame->descriptors.rows;
    int N_t = frame->descriptors_right.rows;
    if (N_q == 0 || N_t == 0) return;

    std::vector<float> y_q(N_q), y_t(N_t), x_q(N_q), x_t(N_t);
    for (int i = 0; i < N_q; ++i) {
        y_q[i] = frame->keypoints[i].pt.y;
        x_q[i] = frame->keypoints[i].pt.x;
    }
    for (int i = 0; i < N_t; ++i) {
        y_t[i] = frame->keypoints_right[i].pt.y;
        x_t[i] = frame->keypoints_right[i].pt.x;
    }

    cv::Mat q = frame->descriptors.isContinuous()       ? frame->descriptors       : frame->descriptors.clone();
    cv::Mat t = frame->descriptors_right.isContinuous() ? frame->descriptors_right : frame->descriptors_right.clone();

    std::vector<int> best_idx(N_q, -1), best_dist(N_q, kMaxHamming);
    cuda_match_stereo_epipolar(
        q.data, t.data, N_q, N_t,
        y_q.data(), y_t.data(), x_q.data(), x_t.data(),
        cfg_.stereo_epi_tol, cfg_.stereo_d_min, cfg_.stereo_d_max,
        cfg_.lowe_ratio, best_idx.data(), best_dist.data());

    for (int i = 0; i < N_q; ++i) {
        if (best_idx[i] >= 0 && best_dist[i] <= cfg_.hamming_threshold) {
            frame->uR[i] = frame->keypoints_right[best_idx[i]].pt.x;
        }
    }
}

// stereo triangulation: Z = fx*b/d

int Tracker::triangulate_stereo(Frame::Ptr frame)
{
    if (frame->uR.empty()) return 0;

    // build a coarse pixel grid of live KLT flow tracks so we can suppress
    // duplicate stereo points within `flow_dup_radius_px` of an existing track
    const float dup_r  = cfg_.flow_dup_radius_px;
    const float dup_r2 = dup_r * dup_r;
    const int   cell   = std::max(4, (int)dup_r);
    const int   gx_n   = (cam_.width  + cell - 1) / cell;
    const int   gy_n   = (cam_.height + cell - 1) / cell;
    std::vector<std::vector<int>> flow_grid;
    if (cam_.width > 0 && !frame->flow_pts.empty()) {
        flow_grid.assign(gx_n * gy_n, {});
        for (int k = 0; k < (int)frame->flow_pts.size(); ++k) {
            const auto& p = frame->flow_pts[k];
            int gx = std::min(gx_n - 1, std::max(0, int(p.x / cell)));
            int gy = std::min(gy_n - 1, std::max(0, int(p.y / cell)));
            flow_grid[gy * gx_n + gx].push_back(k);
        }
    }
    auto near_existing_track = [&](float x, float y) {
        if (flow_grid.empty()) return false;
        int gx = std::min(gx_n - 1, std::max(0, int(x / cell)));
        int gy = std::min(gy_n - 1, std::max(0, int(y / cell)));
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int cx = gx + dx, cy = gy + dy;
                if (cx < 0 || cy < 0 || cx >= gx_n || cy >= gy_n) continue;
                for (int k : flow_grid[cy * gx_n + cx]) {
                    float ddx = frame->flow_pts[k].x - x;
                    float ddy = frame->flow_pts[k].y - y;
                    if (ddx*ddx + ddy*ddy < dup_r2) return true;
                }
            }
        return false;
    };

    int n_added = 0;
    int n_skip_dup = 0;
    for (int i = 0; i < (int)frame->keypoints.size(); ++i) {
        if (frame->uR[i] < 0.0f) continue;  // no stereo match
        if (frame->map_points[i]) continue;  // already mapped
        if (near_existing_track(frame->keypoints[i].pt.x, frame->keypoints[i].pt.y)) {
            ++n_skip_dup; continue;
        }

        float u_L = frame->keypoints[i].pt.x;
        float v_L = frame->keypoints[i].pt.y;
        float u_R = frame->uR[i];
        float d   = u_L - u_R;
        if (d < cfg_.stereo_d_min || d > cfg_.stereo_d_max) continue;

        double Z = cam_.fx * cam_.baseline / (double)d;
        double X = ((double)u_L - cam_.cx) * Z / cam_.fx;
        double Y = ((double)v_L - cam_.cy) * Z / cam_.fy;
        if (Z < 0.5 || Z > cfg_.max_depth_baseline_ratio * cam_.baseline) continue;

        Eigen::Vector3d Xw = frame->T_cw.inverse() * Eigen::Vector3d(X, Y, Z);

        auto mp = MapPoint::create(Xw, g_point_id++);
        mp->observed_times = 2;  // stereo counts as two-view constraint
        frame->map_points[i] = mp;
        map_->insert_map_point(mp);
        ++n_added;
    }
    if (n_skip_dup > 0)
        fprintf(stderr, "[stereo-tri] suppressed %d dup pts near live KLT tracks\n", n_skip_dup);
    return n_added;
}

// after BA corrects KF poses, the inter-frame velocity derived from the last
// tracking step may be inconsistent. Mark it invalid so the next frame falls
// back to "last pose" as the initial guess (safe; one frame without prediction)
void Tracker::notify_ba_update() {
    velocity_valid_ = false;
}

}  // namespace slam

