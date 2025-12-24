#include "slam/map_point.hpp"
#include "slam/frame.hpp"
#include <limits>

namespace slam {

std::atomic<long> MapPoint::next_id_{0};

MapPoint::Ptr MapPoint::create(const Eigen::Vector3d& position, long id)
{
    auto mp = std::shared_ptr<MapPoint>(new MapPoint());
    mp->id       = id;
    mp->position = position;
    return mp;
}

void MapPoint::add_observation(long frame_id, int kp_idx)
{
    std::lock_guard<std::mutex> lock(obs_mutex);
    observations[frame_id] = kp_idx;
    ++observed_times;
}

void MapPoint::remove_observation(long frame_id)
{
    std::lock_guard<std::mutex> lock(obs_mutex);
    observations.erase(frame_id);
}

int MapPoint::get_keypoint_idx(long frame_id) const
{
    std::lock_guard<std::mutex> lock(obs_mutex);
    auto it = observations.find(frame_id);
    return (it != observations.end()) ? it->second : -1;
}

int MapPoint::num_observations() const
{
    std::lock_guard<std::mutex> lock(obs_mutex);
    return static_cast<int>(observations.size());
}

void MapPoint::update_descriptor(const std::vector<std::shared_ptr<Frame>>& frames)
{
    // collect all observed descriptors
    std::vector<cv::Mat> descs;
    {
        std::lock_guard<std::mutex> lock(obs_mutex);
        for (auto& [fid, kp_idx] : observations) {
            for (auto& f : frames) {
                if (f->id == fid && kp_idx < f->descriptors.rows) {
                    descs.push_back(f->descriptors.row(kp_idx));
                    break;
                }
            }
        }
    }

    if (descs.empty()) return;

    // Choose the descriptor with minimum total Hamming distance to all others
    // (the "median" descriptor in Hamming space)
    int best_idx = 0;
    int best_sum = std::numeric_limits<int>::max();
    for (int i = 0; i < (int)descs.size(); ++i) {
        int sum = 0;
        for (int j = 0; j < (int)descs.size(); ++j) {
            if (i != j)
                sum += static_cast<int>(cv::norm(descs[i], descs[j], cv::NORM_HAMMING));
        }
        if (sum < best_sum) {
            best_sum = sum;
            best_idx = i;
        }
    }
    descriptor = descs[best_idx].clone();
}

}  // namespace slam
