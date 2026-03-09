#include "slam/frame.hpp"
#include "slam/map_point.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace slam {

Frame::Ptr Frame::create(const cv::Mat& image, double timestamp, long id)
{
    auto f = std::shared_ptr<Frame>(new Frame());
    f->id        = id;
    f->timestamp = timestamp;

    if (image.channels() == 3) {
        cv::cvtColor(image, f->image_gray, cv::COLOR_BGR2GRAY);
    } else {
        f->image_gray = image.clone();
    }

    // build KLT image pyramid once. cv::calcOpticalFlowPyrLK reuses this directly,
    // avoiding the redundant pyramid build that PyrLK would otherwise do every call
    if (!f->image_gray.empty()) {
        cv::buildOpticalFlowPyramid(f->image_gray, f->klt_pyramid,
                                    cv::Size(21, 21), /*maxLevel=*/3,
                                    /*withDerivatives=*/true);
    }

    return f;
}

int Frame::num_tracked() const
{
    int count = 0;
    for (const auto& mp : map_points) {
        if (mp && !mp->is_bad) ++count;
    }
    return count;
}

}  // namespace slam
