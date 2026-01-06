#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <opencv2/features2d.hpp>
#include <memory>

namespace slam {

enum class TrackingState {
    NOT_INITIALIZED,
    OK,
    LOST
};

// front-end tracker
//   1. stereo init, triangulate metric pts from single frame
//   2. KLT flow w/ fwd-backward consistency
//   3. motion-only ceres BA on surviving 2D-3D matches
//   4. KF insertion on translation/rotation/coverage triggers
//   5. ORB extraction + stereo epipolar match at KFs
class Tracker {
public:
    struct Config {
        int   orb_features       = 2000;
        float orb_scale_factor   = 1.2f;
        int   orb_levels         = 8;
        int   orb_edge_threshold = 31;
        int   hamming_threshold  = 60;
        float lowe_ratio         = 0.75f;

        int   min_tracked_points = 80;
        int   pnp_iterations     = 500;
        float pnp_reprojection   = 4.0f;
        int   pnp_min_inliers    = 20;
        float stereo_epi_tol     = 2.0f;
        float stereo_d_min       = 3.0f;
        float stereo_d_max       = 300.0f;

        // KLT
        int   klt_win            = 21;
        int   klt_max_level      = 3;
        float klt_fb_max_err     = 1.0f;  // fwd-backward threshold px

        // KF trigger AABB fraction
        float kf_min_aabb_frac   = 0.30f;

        // cap flow tracks to bound BA cost, older tracks survive first
        // since they already passed multiple BA cycles
        int   max_flow_tracks    = 800;
        int   flow_grid_cols     = 20;
        int   flow_grid_rows     = 10;
        // suppresion radius to avoid duplicate map points near existing KLT tracks
        float flow_dup_radius_px = 2.5f;

        // motion-only BA
        int    moba_iterations   = 8;
        double moba_huber        = 1.5;

        // geometric KF triggers
        float kf_min_translation = 5.0f;   // metres
        float kf_min_rotation    = 0.12f;  // radians ~6.9 deg
        int   kf_max_frames      = 30;

        // reject stereo pts deeper than ratio * baseline (~43m for KITTI)
        float max_depth_baseline_ratio = 80.0f;
    };

    using Ptr = std::shared_ptr<Tracker>;

    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    bool track(Frame::Ptr frame);
    TrackingState state() const { return state_; }

    // call after BA to re-derive velocity from corrected KF poses
    void notify_ba_update();

private:
    bool initialize(Frame::Ptr frame);
    bool track_with_klt(Frame::Ptr frame);
    bool track_local_map(Frame::Ptr frame);
    bool need_new_keyframe(Frame::Ptr frame) const;
    void insert_keyframe(Frame::Ptr frame);

    // ORB detect+describe on L and R, only at KFs
    void extract_kf_features(Frame::Ptr frame);

    void log_anms_grid_stats(const std::vector<cv::KeyPoint>& kps,
                             int img_w, int img_h,
                             int grid_cols, int grid_rows) const;

    // GPU hamming matcher w/ optional lowe ratio
    std::vector<cv::DMatch> match_descriptors(
        const cv::Mat& query_desc,
        const cv::Mat& train_desc,
        bool use_ratio = true
    );

    int triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                            const std::vector<cv::DMatch>& matches);

    void match_stereo(Frame::Ptr frame);        // GPU stereo epipolar -> fills uR
    int  triangulate_stereo(Frame::Ptr frame);  // Z = fx*b/d

    // bucket flow into spatial grid and trim to max
    void cap_flow_tracks(Frame::Ptr frame) const;
    bool try_relocalize(Frame::Ptr frame);

    double compute_median_parallax(Frame::Ptr frame, Frame::Ptr ref_kf) const;

    Camera         cam_;
    Map::Ptr       map_;
    Config         cfg_;
    TrackingState  state_ = TrackingState::NOT_INITIALIZED;

    cv::Ptr<cv::ORB> orb_;

    Frame::Ptr last_frame_;
    Frame::Ptr last_keyframe_;

    int last_kf_pnp_tracked_ = 0;

    // constant-velocity model: velocity_ = T_cw(t) * T_cw(t-1)^-1
    // used as initial guess before MoBA
    Eigen::Isometry3d velocity_ = Eigen::Isometry3d::Identity();
    bool velocity_valid_ = false;

    int frames_since_kf_ = 0;
};

}  // namespace slam
