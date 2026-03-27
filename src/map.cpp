#include "slam/map.hpp"
#include <algorithm>

namespace slam {

// keyframe management

void Map::insert_keyframe(Frame::Ptr kf)
{
    {
        std::lock_guard<std::mutex> lock(kf_mutex_);
        kf->is_keyframe = true;
        keyframes_[kf->id] = kf;
        keyframe_order_.push_back(kf);
    }
    for (auto& mp : kf->map_points) { // update covis for all pts this kf sees
        if (mp && !mp->is_bad) {
            update_covisibility(kf->id, mp);
        }
    }
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

// map points

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
        for (auto& kf : keyframe_order_) {
            // strip heavy data before archiving (only T_cw needed for viz)
            kf->image_gray.release();
            kf->image_right.release();
            kf->klt_pyramid.clear();
            kf->klt_pyramid_right.clear();
            kf->descriptors.release();
            kf->descriptors_right.release();
            kf->keypoints.clear();
            kf->keypoints_right.clear();
            kf->map_points.clear();
            kf->flow_pts.clear();
            kf->flow_mps.clear();
            kf->uR.clear();
            trajectory_archive_.push_back(kf);
        }
        keyframe_order_.clear();
        keyframes_.clear();
    }
    {
        std::lock_guard<std::mutex> lk(mp_mutex_);
        map_points_.clear();
    }
    {
        std::lock_guard<std::mutex> lk(covis_mutex_);
        covis_.clear();
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

    // archive for traj viz, strip heavy data after
    trajectory_archive_.push_back(oldest);

    // remove observations from oldest KF's points, mark bad if orphaned
    for (int i = 0; i < (int)oldest->map_points.size(); ++i) {
        auto& mp = oldest->map_points[i];
        if (!mp || mp->is_bad) continue;
        mp->remove_observation(oldest->id);
        if (mp->num_observations() == 0) {
            mp->is_bad = true;
        }
    }

    // release heavy data, only need T_cw + id for traj archive
    oldest->image_gray.release();
    oldest->image_right.release();
    oldest->klt_pyramid.clear();
    oldest->klt_pyramid.shrink_to_fit();
    oldest->klt_pyramid_right.clear();
    oldest->klt_pyramid_right.shrink_to_fit();
    oldest->descriptors.release();
    oldest->descriptors_right.release();
    oldest->keypoints.clear();
    oldest->keypoints.shrink_to_fit();
    oldest->keypoints_right.clear();
    oldest->keypoints_right.shrink_to_fit();
    oldest->map_points.clear();
    oldest->map_points.shrink_to_fit();
    oldest->flow_pts.clear();
    oldest->flow_pts.shrink_to_fit();
    oldest->flow_mps.clear();
    oldest->flow_mps.shrink_to_fit();
    oldest->uR.clear();
    oldest->uR.shrink_to_fit();

    // remove KF from map structures
    {
        std::lock_guard<std::mutex> lk(kf_mutex_);
        keyframes_.erase(oldest->id);
        keyframe_order_.pop_front();
    }

    // clean covis edges referencing this KF
    {
        std::lock_guard<std::mutex> lk(covis_mutex_);
        covis_.erase(oldest->id);
        for (auto& [_, neighbors] : covis_)
            neighbors.erase(oldest->id);
    }

    cleanup_bad_map_points();
}

std::vector<Frame::Ptr> Map::trajectory_archive() const {
    return trajectory_archive_;
}

int Map::count_shared_map_points(long kf_id_a, long kf_id_b) const
{
    // covis graph lookup first , avoids O(N) scan
    {
        std::lock_guard<std::mutex> lock(covis_mutex_);
        auto it_a = covis_.find(kf_id_a);
        if (it_a != covis_.end()) {
            auto it_b = it_a->second.find(kf_id_b);
            if (it_b != it_a->second.end()) return it_b->second;
        }
    }
    // fallback brute-force , slow, shouldn't hit this often
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

// covisibility graph

void Map::update_covisibility(long kf_id, MapPoint::Ptr mp)
{
    if (!mp || mp->is_bad) return;

    std::vector<long> other_kfs; // other kfs that also see this point
    {
        std::lock_guard<std::mutex> obs_lock(mp->obs_mutex);
        for (auto& [fid, _] : mp->observations) {
            if (fid != kf_id) other_kfs.push_back(fid);
        }
    }

    if (other_kfs.empty()) return;

    std::lock_guard<std::mutex> lock(covis_mutex_);
    for (long other_id : other_kfs) {
        covis_[kf_id][other_id]++;
        covis_[other_id][kf_id]++;
    }
}

std::vector<std::pair<Frame::Ptr, int>> Map::get_covisible_keyframes(
    long kf_id, int top_n) const
{
    std::vector<std::pair<long, int>> neighbors;
    {
        std::lock_guard<std::mutex> lock(covis_mutex_);
        auto it = covis_.find(kf_id);
        if (it != covis_.end()) {
            neighbors.reserve(it->second.size());
            for (auto& [nid, count] : it->second) {
                neighbors.push_back({nid, count});
            }
        }
    }

    std::sort(neighbors.begin(), neighbors.end(),
              [](auto& a, auto& b) { return a.second > b.second; }); // descending covis count

    std::vector<std::pair<Frame::Ptr, int>> result;
    {
        std::lock_guard<std::mutex> lock(kf_mutex_);
        for (auto& [nid, count] : neighbors) {
            if ((int)result.size() >= top_n) break;
            auto kit = keyframes_.find(nid);
            if (kit != keyframes_.end()) {
                result.push_back({kit->second, count});
            }
        }
    }
    return result;
}

std::vector<Frame::Ptr> Map::get_covisible_keyframes_above(
    long kf_id, int min_shared) const
{
    std::vector<std::pair<long, int>> neighbors;
    {
        std::lock_guard<std::mutex> lock(covis_mutex_);
        auto it = covis_.find(kf_id);
        if (it != covis_.end()) {
            for (auto& [nid, count] : it->second) {
                if (count >= min_shared) neighbors.push_back({nid, count});
            }
        }
    }

    std::sort(neighbors.begin(), neighbors.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    std::vector<Frame::Ptr> result;
    {
        std::lock_guard<std::mutex> lock(kf_mutex_);
        for (auto& [nid, count] : neighbors) {
            auto kit = keyframes_.find(nid);
            if (kit != keyframes_.end()) {
                result.push_back(kit->second);
            }
        }
    }
    return result;
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
