#include "slam/map.hpp"
#include <algorithm>

namespace slam {

void Map::insert_keyframe(Frame::Ptr kf)
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    kf->is_keyframe = true;
    keyframes_[kf->id] = kf;
    keyframe_order_.push_back(kf);
}

void Map::remove_keyframe(long id)
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    keyframes_.erase(id);
    auto it = std::find_if(keyframe_order_.begin(), keyframe_order_.end(),
                           [id](const Frame::Ptr& f) { return f->id == id; });
    if (it != keyframe_order_.end()) keyframe_order_.erase(it);
}

Frame::Ptr Map::get_keyframe(long id) const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    auto it = keyframes_.find(id);
    return (it != keyframes_.end()) ? it->second : nullptr;
}

std::vector<Frame::Ptr> Map::all_keyframes() const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    return std::vector<Frame::Ptr>(keyframe_order_.begin(), keyframe_order_.end());
}

std::vector<Frame::Ptr> Map::local_window() const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    int n = static_cast<int>(keyframe_order_.size());
    int start = std::max(0, n - kWindowSize);
    return std::vector<Frame::Ptr>(
        keyframe_order_.begin() + start, keyframe_order_.end());
}

std::vector<Frame::Ptr> Map::local_window(int size) const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    int n = static_cast<int>(keyframe_order_.size());
    int start = std::max(0, n - size);
    return std::vector<Frame::Ptr>(
        keyframe_order_.begin() + start, keyframe_order_.end());
}

void Map::insert_map_point(MapPoint::Ptr mp)
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    map_points_[mp->id] = mp;
}

void Map::remove_map_point(long id)
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    map_points_.erase(id);
}

MapPoint::Ptr Map::get_map_point(long id) const
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    auto it = map_points_.find(id);
    return (it != map_points_.end()) ? it->second : nullptr;
}

std::vector<MapPoint::Ptr> Map::all_map_points() const
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    std::vector<MapPoint::Ptr> out;
    out.reserve(map_points_.size());
    for (auto& [id, mp] : map_points_) {
        if (!mp->is_bad) out.push_back(mp);
    }
    return out;
}

void Map::cleanup_bad_map_points()
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    for (auto it = map_points_.begin(); it != map_points_.end(); ) {
        if (it->second->is_bad) it = map_points_.erase(it);
        else                    ++it;
    }
}

void Map::reset()
{
    {
        std::lock_guard<std::mutex> lk(kf_mutex_);
        keyframe_order_.clear();
        keyframes_.clear();
    }
    {
        std::lock_guard<std::mutex> lk(mp_mutex_);
        map_points_.clear();
    }
}

void Map::evict_oldest_keyframe()
{
    Frame::Ptr oldest;
    {
        std::lock_guard<std::mutex> lk(kf_mutex_);
        if (keyframe_order_.empty()) return;
        oldest = keyframe_order_.front();
    }

    for (int i = 0; i < (int)oldest->map_points.size(); ++i) {
        auto& mp = oldest->map_points[i];
        if (!mp || mp->is_bad) continue;
        mp->remove_observation(oldest->id);
        if (mp->num_observations() == 0) {
            mp->is_bad = true;
        }
    }

    oldest->image_gray.release();
    oldest->descriptors.release();
    oldest->keypoints.clear();
    oldest->keypoints.shrink_to_fit();
    oldest->map_points.clear();
    oldest->map_points.shrink_to_fit();

    {
        std::lock_guard<std::mutex> lk(kf_mutex_);
        keyframes_.erase(oldest->id);
        keyframe_order_.pop_front();
    }

    cleanup_bad_map_points();
}

int Map::count_shared_map_points(long kf_id_a, long kf_id_b) const
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    int count = 0;
    for (auto& [id, mp] : map_points_) {
        if (mp->is_bad) continue;
        std::lock_guard<std::mutex> obs_lock(mp->obs_mutex);
        if (mp->observations.count(kf_id_a) && mp->observations.count(kf_id_b))
            ++count;
    }
    return count;
}

size_t Map::num_keyframes() const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);
    return keyframes_.size();
}

size_t Map::num_map_points() const
{
    std::lock_guard<std::mutex> lock(mp_mutex_);
    return map_points_.size();
}

}  // namespace slam
