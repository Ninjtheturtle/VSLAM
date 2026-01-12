// sliding-window bundle adjustment w/ Ceres
// quaternion+translation parameterization (7-DOF), analytical Jacobians
// information-weighted stereo residuals w/ depth-attenuated disparity term
// oldest KF locked (all 6 DOFs) for gauge; clean eviction when window fills

#include "slam/local_ba.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace slam {

// stereo reprojection cost , 3 residuals (u_L, v_L, u_R), analytical jacobians
// SizedCostFunction<3, 7, 3>

StereoReprojCost::StereoReprojCost(double obs_uL, double obs_vL, double obs_uR, double fx,
                                   double fy, double cx, double cy, double b, double info_uv,
                                   double info_disp)
    : obs_uL_(obs_uL),
      obs_vL_(obs_vL),
      obs_uR_(obs_uR),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      b_(b),
      sqrt_info_uv_(std::sqrt(std::max(info_uv, 1e-6))),
      sqrt_info_d_(std::sqrt(std::max(info_disp, 1e-6))) {}

bool StereoReprojCost::Evaluate(double const* const* parameters, double* residuals,
                                double** jacobians) const {
    // params[0] = pose [qx,qy,qz,qw, tx,ty,tz] , Eigen stores (x,y,z,w) internally
    // params[1] = point [X,Y,Z]
    Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> Pw(parameters[1]);

    // forward pass: Pc = R(q)*Pw + t
    const Eigen::Matrix3d R = q.toRotationMatrix();
    const Eigen::Vector3d Pc = R * Pw + t;

    const double Xc = Pc.x(), Yc = Pc.y(), Zc = Pc.z();
    const double inv_z = 1.0 / Zc;
    const double inv_z2 = inv_z * inv_z;

    // information-weighted residuals
    //   r0 = sqrt_Omega_uv * (fx*Xc/Zc + cx - uL)
    //   r1 = sqrt_Omega_uv * (fy*Yc/Zc + cy - vL)
    //   r2 = sqrt_Omega_d  * (fx*(Xc-b)/Zc + cx - uR)
    residuals[0] = sqrt_info_uv_ * (fx_ * Xc * inv_z + cx_ - obs_uL_);
    residuals[1] = sqrt_info_uv_ * (fy_ * Yc * inv_z + cy_ - obs_vL_);
    residuals[2] = sqrt_info_d_ * (fx_ * (Xc - b_) * inv_z + cx_ - obs_uR_);

    if (!jacobians) return true;

    // projection Jacobian J_proj (3x3) , sparse, only 5 nonzero entries
    const double j00 = fx_ * inv_z;
    const double j02 = -fx_ * Xc * inv_z2;
    const double j11 = fy_ * inv_z;
    const double j12 = -fy_ * Yc * inv_z2;
    const double j22 = -fx_ * (Xc - b_) * inv_z2;

    // pose Jacobian (3x7): chain rule through dR/dq_i * Pw
    if (jacobians[0]) {
        // each dR/dq_i is a 3x3 matrix; we only need its product w/ Pw
        // these are the four d(R*Pw)/dq_i 3-vectors

        const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();
        const double X = Pw.x(), Y = Pw.y(), Z = Pw.z();

        const double d0_qx = 2.0 * (qy * Y + qz * Z);
        const double d1_qx = 2.0 * (qy * X - 2.0 * qx * Y - qw * Z);
        const double d2_qx = 2.0 * (qz * X + qw * Y - 2.0 * qx * Z);

        const double d0_qy = 2.0 * (-2.0 * qy * X + qx * Y + qw * Z);
        const double d1_qy = 2.0 * (qx * X + qz * Z);
        const double d2_qy = 2.0 * (-qw * X + qz * Y - 2.0 * qy * Z);

        const double d0_qz = 2.0 * (-2.0 * qz * X - qw * Y + qx * Z);
        const double d1_qz = 2.0 * (qw * X - 2.0 * qz * Y + qy * Z);
        const double d2_qz = 2.0 * (qx * X + qy * Y);

        const double d0_qw = 2.0 * (-qz * Y + qy * Z);
        const double d1_qw = 2.0 * (qz * X - qx * Z);
        const double d2_qw = 2.0 * (-qy * X + qx * Y);

        // J[row][col] = sqrt_info * J_proj[row] dot dPc_dq[col]
        double* J = jacobians[0];  // row-major 3x7

        // row 0 (u_L)
        J[0] = sqrt_info_uv_ * (j00 * d0_qx + j02 * d2_qx);
        J[1] = sqrt_info_uv_ * (j00 * d0_qy + j02 * d2_qy);
        J[2] = sqrt_info_uv_ * (j00 * d0_qz + j02 * d2_qz);
        J[3] = sqrt_info_uv_ * (j00 * d0_qw + j02 * d2_qw);

        // row 1 (v_L)
        J[7] = sqrt_info_uv_ * (j11 * d1_qx + j12 * d2_qx);
        J[8] = sqrt_info_uv_ * (j11 * d1_qy + j12 * d2_qy);
        J[9] = sqrt_info_uv_ * (j11 * d1_qz + j12 * d2_qz);
        J[10] = sqrt_info_uv_ * (j11 * d1_qw + j12 * d2_qw);

        // row 2 (u_R)
        J[14] = sqrt_info_d_ * (j00 * d0_qx + j22 * d2_qx);
        J[15] = sqrt_info_d_ * (j00 * d0_qy + j22 * d2_qy);
        J[16] = sqrt_info_d_ * (j00 * d0_qz + j22 * d2_qz);
        J[17] = sqrt_info_d_ * (j00 * d0_qw + j22 * d2_qw);

        // translation cols 4-6: dPc/dt = I3, so just J_proj directly
        J[4] = sqrt_info_uv_ * j00;
        J[5] = 0.0;
        J[6] = sqrt_info_uv_ * j02;
        J[11] = 0.0;
        J[12] = sqrt_info_uv_ * j11;
        J[13] = sqrt_info_uv_ * j12;
        J[18] = sqrt_info_d_ * j00;
        J[19] = 0.0;
        J[20] = sqrt_info_d_ * j22;
    }

    // point Jacobian (3x3): dPc/dPw = R, so J_point = sqrt_info * J_proj * R
    if (jacobians[1]) {
        double* J = jacobians[1];  // row-major 3x3
        for (int c = 0; c < 3; ++c) {
            J[c] = sqrt_info_uv_ * (j00 * R(0, c) + j02 * R(2, c));
            J[3 + c] = sqrt_info_uv_ * (j11 * R(1, c) + j12 * R(2, c));
            J[6 + c] = sqrt_info_d_ * (j00 * R(0, c) + j22 * R(2, c));
        }
    }

    return true;
}

// mono reprojection cost , 2 residuals (u, v)
// SizedCostFunction<2, 7, 3>
// same math as stereo but without the right-camera residual

MonoReprojCost::MonoReprojCost(double obs_u, double obs_v, double fx, double fy, double cx,
                               double cy, double info_uv)
    : obs_u_(obs_u),
      obs_v_(obs_v),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      sqrt_info_uv_(std::sqrt(std::max(info_uv, 1e-6))) {}

bool MonoReprojCost::Evaluate(double const* const* parameters, double* residuals,
                              double** jacobians) const {
    Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> Pw(parameters[1]);

    const Eigen::Matrix3d R = q.toRotationMatrix();
    const Eigen::Vector3d Pc = R * Pw + t;

    const double Xc = Pc.x(), Yc = Pc.y(), Zc = Pc.z();
    const double inv_z = 1.0 / Zc;
    const double inv_z2 = inv_z * inv_z;

    residuals[0] = sqrt_info_uv_ * (fx_ * Xc * inv_z + cx_ - obs_u_);
    residuals[1] = sqrt_info_uv_ * (fy_ * Yc * inv_z + cy_ - obs_v_);

    if (!jacobians) return true;

    const double j00 = fx_ * inv_z;
    const double j02 = -fx_ * Xc * inv_z2;
    const double j11 = fy_ * inv_z;
    const double j12 = -fy_ * Yc * inv_z2;

    if (jacobians[0]) {
        const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();
        const double X = Pw.x(), Y = Pw.y(), Z = Pw.z();

        const double d0_qx = 2.0 * (qy * Y + qz * Z);
        const double d1_qx = 2.0 * (qy * X - 2.0 * qx * Y - qw * Z);
        const double d2_qx = 2.0 * (qz * X + qw * Y - 2.0 * qx * Z);

        const double d0_qy = 2.0 * (-2.0 * qy * X + qx * Y + qw * Z);
        const double d1_qy = 2.0 * (qx * X + qz * Z);
        const double d2_qy = 2.0 * (-qw * X + qz * Y - 2.0 * qy * Z);

        const double d0_qz = 2.0 * (-2.0 * qz * X - qw * Y + qx * Z);
        const double d1_qz = 2.0 * (qw * X - 2.0 * qz * Y + qy * Z);
        const double d2_qz = 2.0 * (qx * X + qy * Y);

        const double d0_qw = 2.0 * (-qz * Y + qy * Z);
        const double d1_qw = 2.0 * (qz * X - qx * Z);
        const double d2_qw = 2.0 * (-qy * X + qx * Y);

        double* J = jacobians[0];  // row-major 2x7

        J[0] = sqrt_info_uv_ * (j00 * d0_qx + j02 * d2_qx);
        J[1] = sqrt_info_uv_ * (j00 * d0_qy + j02 * d2_qy);
        J[2] = sqrt_info_uv_ * (j00 * d0_qz + j02 * d2_qz);
        J[3] = sqrt_info_uv_ * (j00 * d0_qw + j02 * d2_qw);

        J[7] = sqrt_info_uv_ * (j11 * d1_qx + j12 * d2_qx);
        J[8] = sqrt_info_uv_ * (j11 * d1_qy + j12 * d2_qy);
        J[9] = sqrt_info_uv_ * (j11 * d1_qz + j12 * d2_qz);
        J[10] = sqrt_info_uv_ * (j11 * d1_qw + j12 * d2_qw);

        J[4] = sqrt_info_uv_ * j00;
        J[5] = 0.0;
        J[6] = sqrt_info_uv_ * j02;
        J[11] = 0.0;
        J[12] = sqrt_info_uv_ * j11;
        J[13] = sqrt_info_uv_ * j12;
    }

    if (jacobians[1]) {
        double* J = jacobians[1];  // row-major 2x3
        for (int c = 0; c < 3; ++c) {
            J[c] = sqrt_info_uv_ * (j00 * R(0, c) + j02 * R(2, c));
            J[3 + c] = sqrt_info_uv_ * (j11 * R(1, c) + j12 * R(2, c));
        }
    }

    return true;
}

// pose <-> Isometry3d conversion (quaternion 7-DOF)
static void isometry_to_pose(const Eigen::Isometry3d& T, double* pose) {
    Eigen::Quaterniond q(T.rotation());
    q.normalize();
    // Eigen coeffs() = (x,y,z,w)
    pose[0] = q.x();
    pose[1] = q.y();
    pose[2] = q.z();
    pose[3] = q.w();
    pose[4] = T.translation().x();
    pose[5] = T.translation().y();
    pose[6] = T.translation().z();
}

static Eigen::Isometry3d pose_to_isometry(const double* pose) {
    Eigen::Quaterniond q(pose[3], pose[0], pose[1], pose[2]);  // ctor is (w,x,y,z)
    q.normalize();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() << pose[4], pose[5], pose[6];
    return T;
}

// quaternion point rotation for post-BA reprojection culling
// hamilton double-cross: result = q * p * q^{-1}
static void quat_rotate_point(const double* q, const double* p, double* result) {
    const double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    // t = 2*(v x p)
    const double tx = 2.0 * (qy * p[2] - qz * p[1]);
    const double ty = 2.0 * (qz * p[0] - qx * p[2]);
    const double tz = 2.0 * (qx * p[1] - qy * p[0]);
    // result = p + w*t + v x t
    result[0] = p[0] + qw * tx + qy * tz - qz * ty;
    result[1] = p[1] + qw * ty + qz * tx - qx * tz;
    result[2] = p[2] + qw * tz + qx * ty - qy * tx;
}

// local BA

LocalBA::Ptr LocalBA::create(const Camera& cam, Map::Ptr map, const Config& cfg) {
    auto ba = std::shared_ptr<LocalBA>(new LocalBA());
    ba->cam_ = cam;
    ba->map_ = map;
    ba->cfg_ = cfg;
    return ba;
}

void LocalBA::optimize() {
    auto window = map_->local_window();
    if (window.size() < 2) return;

    // collect all map pts visible in window
    std::unordered_map<long, MapPoint::Ptr> active_points;
    for (auto& kf : window) {
        for (auto& mp : kf->map_points) {
            if (mp && !mp->is_bad) {
                active_points[mp->id] = mp;
            }
        }
    }
    if (active_points.empty()) return;

    // 7 doubles per pose: [qx,qy,qz,qw, tx,ty,tz]
    std::unordered_map<long, std::vector<double>> pose_params;
    for (auto& kf : window) {
        pose_params[kf->id].resize(7);
        isometry_to_pose(kf->T_cw, pose_params[kf->id].data());
    }

    std::unordered_map<long, std::array<double, 3>> point_params;
    for (auto& [id, mp] : active_points) {
        point_params[id] = {mp->position.x(), mp->position.y(), mp->position.z()};
    }

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(cfg_.huber_delta);

    // stereo information model:
    //   Omega_uv = 1/sigma^2                    (bearing , constant)
    //   Omega_d  = (1/sigma^2) * min(1, Z_ref^2/Z^2)  (disparity , attenuated at depth)
    // sigma_Z = Z^2*sigma_d/(fx*b) grows quadratic w/ depth
    const double sigma2 = cfg_.sigma_px * cfg_.sigma_px;
    const double info_uv = 1.0 / sigma2;
    const double z_ref2 = cfg_.z_ref * cfg_.z_ref;

    for (auto& kf : window) {
        double* pose = pose_params[kf->id].data();

        Eigen::Quaterniond q_kf(pose[3], pose[0], pose[1], pose[2]);
        q_kf.normalize();
        const Eigen::Matrix3d R_kf = q_kf.toRotationMatrix();
        const Eigen::Vector3d t_kf(pose[4], pose[5], pose[6]);

        for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
            auto& mp = kf->map_points[kp_idx];
            if (!mp || mp->is_bad) continue;

            auto pit = point_params.find(mp->id);
            if (pit == point_params.end()) continue;

            double* pt = pit->second.data();
            const cv::Point2f& obs = kf->keypoints[kp_idx].pt;

            // depth at current linearization pt , used for disparity info weighting
            const Eigen::Vector3d Pw(pt[0], pt[1], pt[2]);
            const double Zc = (R_kf * Pw + t_kf).z();
            if (Zc <= 0.01) continue;  // behind camera

            if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                const double depth_factor = std::min(1.0, z_ref2 / (Zc * Zc));
                const double info_disp = info_uv * depth_factor;

                problem.AddResidualBlock(
                    StereoReprojCost::Create(obs.x, obs.y, kf->uR[kp_idx], cam_.fx, cam_.fy,
                                             cam_.cx, cam_.cy, cam_.baseline, info_uv, info_disp),
                    loss, pose, pt);
            } else {
                problem.AddResidualBlock(MonoReprojCost::Create(obs.x, obs.y, cam_.fx, cam_.fy,
                                                                cam_.cx, cam_.cy, info_uv),
                                         loss, pose, pt);
            }
        }
    }

    // register quaternion manifold on each pose block
    for (auto& kf : window) {
        double* pose = pose_params[kf->id].data();
        problem.AddParameterBlock(pose, 7,
                                  new ceres::ProductManifold(new ceres::EigenQuaternionManifold,
                                                             new ceres::EuclideanManifold<3>));
    }

    for (auto& [id, pt] : point_params) {
        problem.AddParameterBlock(pt.data(), 3);
    }

    // gauge anchor: lock oldest KF
    if (!window.empty()) {
        problem.SetParameterBlockConstant(pose_params[window.front()->id].data());
    }

    // solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = cfg_.max_iterations;
    options.minimizer_progress_to_stdout = cfg_.verbose;
    options.num_threads = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (cfg_.verbose) {
        std::cout << summary.BriefReport() << "\n";
    }

    // write back optimised poses + log yaw correction for diagnostics
    for (auto& kf : window) {
        Eigen::Isometry3d T_old = kf->T_cw;
        kf->T_cw = pose_to_isometry(pose_params[kf->id].data());
        Eigen::Matrix3d R_wc_old = T_old.inverse().rotation();
        Eigen::Matrix3d R_wc_new = kf->T_cw.inverse().rotation();
        double yaw_old =
            std::atan2(R_wc_old(0, 2), R_wc_old(0, 0)) * (180.0 / 3.14159265358979323846);
        double yaw_new =
            std::atan2(R_wc_new(0, 2), R_wc_new(0, 0)) * (180.0 / 3.14159265358979323846);
        double delta_yaw = yaw_new - yaw_old;
        while (delta_yaw > 180.0) delta_yaw -= 360.0;
        while (delta_yaw < -180.0) delta_yaw += 360.0;
        if (std::abs(delta_yaw) > 0.01)
            fprintf(stderr, "[BA-DIAG] kf=%ld delta_yaw=%.4f deg\n", kf->id, delta_yaw);
    }

    // write back optimised 3D positions
    for (auto& [id, mp] : active_points) {
        auto& pt = point_params[id];
        mp->position = Eigen::Vector3d(pt[0], pt[1], pt[2]);
    }

    // cull high-reprojection-error observations
    {
        const double cull_thresh2 = 9.0;  // 3px^2
        for (auto& kf : window) {
            const double* pose = pose_params.at(kf->id).data();
            for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
                auto& mp = kf->map_points[kp_idx];
                if (!mp || mp->is_bad) continue;
                auto pit = point_params.find(mp->id);
                if (pit == point_params.end()) continue;
                const double* pt = pit->second.data();

                double Xc[3];
                quat_rotate_point(pose, pt, Xc);
                Xc[0] += pose[4];
                Xc[1] += pose[5];
                Xc[2] += pose[6];

                if (Xc[2] <= 0.0) {
                    mp->is_bad = true;
                    continue;
                }

                // cull stereo pts beyond max_depth_baseline_ratio × baseline
                if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f &&
                    Xc[2] > cfg_.max_depth_baseline_ratio * cam_.baseline) {
                    mp->is_bad = true;
                    continue;
                }

                // depth-adaptive: LOOSER threshold at distance (far pts have worse depth)
                // at Z=20m scale=1 so 7px, at Z=60m scale=3 so 12px
                double depth_scale = std::max(1.0, Xc[2] / 20.0);
                double adaptive_cull2 = cull_thresh2 * depth_scale;

                double u = cam_.fx * Xc[0] / Xc[2] + cam_.cx;
                double v = cam_.fy * Xc[1] / Xc[2] + cam_.cy;
                double du = u - kf->keypoints[kp_idx].pt.x;
                double dv = v - kf->keypoints[kp_idx].pt.y;
                if (du * du + dv * dv > adaptive_cull2) {
                    mp->is_bad = true;
                    continue;
                }

                // right-camera reproj check
                if (cam_.is_stereo() && kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                    double u_R = cam_.fx * (Xc[0] - cam_.baseline) / Xc[2] + cam_.cx;
                    double dur = u_R - kf->uR[kp_idx];
                    if (dur * dur > adaptive_cull2) mp->is_bad = true;
                }
            }
        }
    }

    // observation-ratio culling: pts frequently visible but rarely matched
    // probably dynamic objects or unstable features
    {
        const int min_visible = 10;
        const float min_ratio = 0.25f;  // need >= 25% match rate
        for (auto& [id, mp] : active_points) {
            if (mp->is_bad) continue;
            if (mp->visible_times >= min_visible) {
                float ratio = (float)mp->observed_times / (float)mp->visible_times;
                if (ratio < min_ratio) {
                    mp->is_bad = true;
                }
            }
        }
    }

    map_->cleanup_bad_map_points();

    // evict oldest KF if window is full
    if ((int)window.size() >= cfg_.window_size) {
        map_->evict_oldest_keyframe();
    }
}

}  // namespace slam
