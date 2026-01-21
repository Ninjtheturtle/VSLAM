#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace rerun { class RecordingStream; }

namespace slam {

// streams SLAM state to rerun via TCP on localhost:9876
class Visualizer {
public:
    struct Config {
        std::string app_id   = "vslam2";
        std::string addr     = "127.0.0.1:9876";
        bool log_image       = true;
        bool log_keypoints   = true;
    };

    using Ptr = std::shared_ptr<Visualizer>;
    static Ptr create(const Config& cfg = Config{});

    ~Visualizer();

    // call once before log_frame so rerun renders the camera frustum
    void log_pinhole(const Camera& cam);

    void log_frame(const Frame::Ptr& frame);
    void log_map(const Map::Ptr& map, double timestamp = 0.0);

    // rebuilt every call from BA-corrected KF poses
    void log_trajectory(const Map::Ptr& map,
                        const Frame::Ptr& current_frame,
                        double ts);

    // static GT trajectory, call once before main loop
    void log_ground_truth(const std::vector<std::array<float, 3>>& centers);

private:
    Visualizer() = default;

    Config cfg_;
    Camera cam_;
    std::unique_ptr<rerun::RecordingStream> rec_;
};

}  // namespace slam
