#include "slam/camera.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace slam {

Camera Camera::from_kitti_calib(const std::string& calib_file)
{
    std::ifstream f(calib_file);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open calib file: " + calib_file);
    }

    Camera cam;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("P0:", 0) == 0) {
            std::istringstream ss(line.substr(3));
            double vals[12];
            for (int i = 0; i < 12; ++i) ss >> vals[i];
            cam.fx = vals[0];
            cam.fy = vals[5];
            cam.cx = vals[2];
            cam.cy = vals[6];
            cam.k1 = 0.0;
            cam.k2 = 0.0;
            cam.width  = 0;
            cam.height = 0;
            break;
        }
    }

    return cam;
}

}  // namespace slam
