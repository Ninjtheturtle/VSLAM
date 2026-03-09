#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <atomic>

namespace slam {

class MapPoint;

// T_cw transforms world -> camera: X_c = R*X_w + t
class Frame {
public:
    using Ptr = std::shared_ptr<Frame>;

    static Ptr create(const cv::Mat& image, double timestamp, long id);

    long   id;
    double timestamp;
    cv::Mat image_gray;

    // KLT pyramid built once at construction, reused by calcOpticalFlowPyrLK
    // on non-KF frames these are the only tracked features
    std::vector<cv::Mat>                   klt_pyramid;
    std::vector<cv::Point2f>               flow_pts;   // current 2D positions
    std::vector<std::shared_ptr<MapPoint>> flow_mps;   // matching 3D map points

    // left image features (KFs only)
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;  // Nx32 CV_8U, ORB

    // right image, stereo only
    cv::Mat                   image_right;
    std::vector<cv::Mat>      klt_pyramid_right;
    std::vector<cv::KeyPoint> keypoints_right;
    cv::Mat                   descriptors_right;

    std::vector<float> uR;  // right x-coord per left kp, -1 = no match

    // one per kp, nullptr = unmatched
    std::vector<std::shared_ptr<MapPoint>> map_points;

    Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();

    bool is_keyframe = false;

    Eigen::Isometry3d T_wc() const { return T_cw.inverse(); }
    Eigen::Vector3d camera_center() const { return T_wc().translation(); }
    const uint8_t* desc_ptr() const { return descriptors.data; }
    int num_features() const { return static_cast<int>(keypoints.size()); }
    int num_tracked() const;

private:
    Frame() = default;
};

}  // namespace slam
