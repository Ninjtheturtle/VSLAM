// rerun logging , 3D entities must live under "world/", NOT "world/camera/" or rerun creates a 2D view

#include "slam/visualizer.hpp"

#include <rerun.hpp>
#include <rerun/archetypes/image.hpp>
#include <rerun/archetypes/pinhole.hpp>
#include <rerun/archetypes/points2d.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/transform3d.hpp>
#include <rerun/archetypes/view_coordinates.hpp>
#include <rerun/components/color.hpp>
#include <rerun/blueprint/archetypes/viewport_blueprint.hpp>

#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <iostream>
#include <vector>

namespace slam {

Visualizer::Ptr Visualizer::create(const Config& cfg)
{
    auto v = std::shared_ptr<Visualizer>(new Visualizer());
    v->cfg_ = cfg;

    v->rec_ = std::make_unique<rerun::RecordingStream>(cfg.app_id);
    auto result = v->rec_->connect_tcp(cfg.addr);
    if (!result.is_ok()) {
        std::cerr << "[Visualizer] Warning: could not connect to Rerun at "
                  << cfg.addr << " — " << result.description << "\n"
                  << "  Start the viewer with:  rerun\n";
    } else {
        std::cout << "[Visualizer] Connected to Rerun at " << cfg.addr << "\n";
    }

    // workaround: reset viewport blueprint so stale cached layouts don't break the 3D view
    {
        rerun::RecordingStream bp(cfg.app_id, "", rerun::StoreKind::Blueprint);
        bp.connect_tcp(cfg.addr);
        bp.log_static("viewport",
            rerun::blueprint::archetypes::ViewportBlueprint()
                .with_auto_layout(true)
                .with_auto_views(true)
                .with_past_viewer_recommendations({}));
    }

    v->rec_->log_static("world", rerun::archetypes::ViewCoordinates::RDF); // KITTI cam convention

    // tiny anchor so rerun creates the 3D view before any real data arrives
    v->rec_->log_static("world/origin",
        rerun::archetypes::Points3D({{0.0f, 0.0f, 0.0f}})
            .with_radii(std::vector<float>{0.001f}));

    return v;
}

Visualizer::~Visualizer() = default;

void Visualizer::log_pinhole(const Camera& cam)
{
    if (!rec_) return;
    cam_ = cam;  // stash for later use in log_frame

    rec_->log_static("world/camera/image",
        rerun::archetypes::Pinhole::from_focal_length_and_resolution(
            {(float)cam.fx, (float)cam.fy},
            {(float)cam.width, (float)cam.height}
        )
    );

    // identity T so the frustum shows before tracking starts
    rec_->log_static("world/camera/image",
        rerun::archetypes::Transform3D::from_translation_rotation(
            {0.0f, 0.0f, 0.0f},
            rerun::datatypes::Quaternion::from_wxyz(1.0f, 0.0f, 0.0f, 0.0f)));
}

void Visualizer::log_frame(const Frame::Ptr& frame)
{
    if (!rec_) return;

    rec_->set_time_seconds("time", frame->timestamp); // enables scrubbing in viewer

    if (frame->num_tracked() > 0) { // move the camera frustum in 3D
        Eigen::Isometry3d  T_wc = frame->T_wc();
        Eigen::Quaterniond q(T_wc.rotation());
        Eigen::Vector3d    t = T_wc.translation();
        rec_->log("world/camera/image",
            rerun::archetypes::Transform3D::from_translation_rotation(
                {(float)t.x(), (float)t.y(), (float)t.z()},
                rerun::datatypes::Quaternion::from_wxyz(
                    (float)q.w(), (float)q.x(), (float)q.y(), (float)q.z())));
    }

    if (cfg_.log_image && !frame->image_gray.empty()) {
        cv::Mat rgb;
        cv::cvtColor(frame->image_gray, rgb, cv::COLOR_GRAY2RGB);
        auto bytes = std::vector<uint8_t>(rgb.data, rgb.data + rgb.total() * 3);
        rec_->log("world/camera/image",
            rerun::archetypes::Image::from_rgb24(
                std::move(bytes), {(uint32_t)rgb.cols, (uint32_t)rgb.rows}));
    }

    if (cfg_.log_keypoints && !frame->keypoints.empty()) {
        std::vector<rerun::datatypes::Vec2D> pts;
        pts.reserve(frame->keypoints.size());
        for (auto& kp : frame->keypoints)
            pts.push_back({kp.pt.x, kp.pt.y});
        rec_->log("world/camera/image/keypoints",
            rerun::archetypes::Points2D(pts)
                .with_colors(rerun::components::Color(190, 75, 230, 220))
                .with_radii(std::vector<float>(pts.size(), 2.5f)));
    }

}

// rebuilds the full trajectory each frame so BA corrections show immediately
// gap > 50m -> new strip segment (handles reinit / LOST jumps)

void Visualizer::log_trajectory(const Map::Ptr& map,
                                 const Frame::Ptr& current_frame,
                                 double ts)
{
    if (!rec_) return;
    rec_->set_time_seconds("time", ts);

    auto archived = map->trajectory_archive();   // kfs from prior map segments , survives reset
    auto kfs      = map->all_keyframes();

    constexpr double kGapSq = 50.0 * 50.0;  // 50m gap threshold , magic but works for KITTI

    std::vector<std::vector<rerun::datatypes::Vec3D>> segments;
    Eigen::Vector3d last_c;
    bool has_last = false;

    auto add_kf_pos = [&](const Frame::Ptr& kf) {
        Eigen::Vector3d c = kf->camera_center();
        if (!has_last || (c - last_c).squaredNorm() > kGapSq) {
            segments.emplace_back();
        }
        segments.back().push_back({(float)c.x(), (float)c.y(), (float)c.z()});
        last_c   = c;
        has_last = true;
    };

    for (auto& kf : archived) add_kf_pos(kf);
    for (auto& kf : kfs)      add_kf_pos(kf);

    // append live frame if not yet promoted to KF , avoids duplicate when KF was just inserted
    if (current_frame && current_frame->num_tracked() > 0 && !current_frame->is_keyframe) {
        Eigen::Vector3d c = current_frame->camera_center();
        if (!has_last || (c - last_c).squaredNorm() > kGapSq) {
            segments.emplace_back();
        }
        if (segments.empty()) segments.emplace_back();
        segments.back().push_back({(float)c.x(), (float)c.y(), (float)c.z()});
    }

    if (segments.empty()) return;

    std::vector<rerun::components::LineStrip3D> strips;
    strips.reserve(segments.size());
    for (auto& seg : segments)
        strips.emplace_back(seg);

    rec_->log("world/camera/trajectory",
        rerun::archetypes::LineStrips3D(strips)
            .with_colors({rerun::components::Color(0, 255, 128)})   // bright green
            .with_radii(std::vector<float>(strips.size(), 0.5f)));
}

void Visualizer::log_map(const Map::Ptr& map, double timestamp)
{
    if (!rec_) return;
    rec_->set_time_seconds("time", timestamp);

    auto map_pts = map->all_map_points();
    if (map_pts.empty()) return;

    std::vector<rerun::datatypes::Vec3D> positions;
    positions.reserve(map_pts.size());

    for (auto& mp : map_pts) {
        auto& p = mp->position;
        positions.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
    }

    rec_->log("world/camera/map/points",
        rerun::archetypes::Points3D(positions)
            .with_radii(std::vector<float>(positions.size(), 0.03f))
    );
}

void Visualizer::log_ground_truth(const std::vector<std::array<float, 3>>& centers)
{
    if (!rec_ || centers.size() < 2) return;

    std::vector<rerun::datatypes::Vec3D> pts;
    pts.reserve(centers.size());
    for (auto& c : centers)
        pts.push_back({c[0], c[1], c[2]});

    rec_->log_static("world/camera/ground_truth",
        rerun::archetypes::LineStrips3D(
            {rerun::components::LineStrip3D(pts)})
            .with_colors({rerun::components::Color(255, 165, 0)})  // orange
            .with_radii({0.5f}));
}

}  // namespace slam
