#pragma once

#include "slam/camera.hpp"
#include "slam/map.hpp"

#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <memory>
#include <cmath>
#include <vector>
#include <array>
#include <unordered_map>

namespace slam {

// sliding-window BA w/ ceres
// last N KF poses + visible map pts, oldest locked for gauge
// 7-DOF [qx,qy,qz,qw,tx,ty,tz] w/ EigenQuaternionManifold
// analytical jacobians for stereo/mono reproj
class LocalBA {
public:
    struct Config {
        int    max_iterations    = 15;
        int    window_size       = Map::kWindowSize;
        double huber_delta       = 1.5;
        double sigma_px          = 1.0;
        double z_ref             = 15.0;  // depth ref for info attenuation
        double max_depth_baseline_ratio = 80.0;  // ~43m for KITTI
        bool   verbose           = false;
    };

    using Ptr = std::shared_ptr<LocalBA>;
    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    void optimize();

private:
    Camera   cam_;
    Map::Ptr map_;
    Config   cfg_;

    LocalBA() = default;
};

// analytical cost functinos (quat parameterized)
// pose: 7 doubles [qx,qy,qz,qw,tx,ty,tz], eigen storage (x,y,z,w)
// point: 3 doubles [X,Y,Z] world frame
// dR/dq_i treating all 4 quat components as independent (ambient derivative)
// ceres EigenQuaternionManifold projects onto 3-DOF tangent plane
//
// info weighting:
//   Omega_uv = 1/sigma^2 (bearing, constant)
//   Omega_d  = Omega_uv * min(1, Z_ref^2/Z^2) (disparity, attenuated at depth)
//   residuals premultiplied by sqrt(Omega_i)

// stereo: 3 residuals (u_L, v_L, u_R)
class StereoReprojCost final : public ceres::SizedCostFunction<3, 7, 3> {
public:
    StereoReprojCost(double obs_uL, double obs_vL, double obs_uR,
                     double fx, double fy, double cx, double cy,
                     double b, double info_uv, double info_disp);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override;

    static ceres::CostFunction* Create(double obs_uL, double obs_vL, double obs_uR,
                                        double fx, double fy, double cx, double cy,
                                        double b, double info_uv, double info_disp) {
        return new StereoReprojCost(obs_uL, obs_vL, obs_uR, fx, fy, cx, cy, b,
                                    info_uv, info_disp);
    }

private:
    double obs_uL_, obs_vL_, obs_uR_;
    double fx_, fy_, cx_, cy_, b_;
    double sqrt_info_uv_;
    double sqrt_info_d_;
};

// mono: 2 residuals (u, v), used when uR < 0
class MonoReprojCost final : public ceres::SizedCostFunction<2, 7, 3> {
public:
    MonoReprojCost(double obs_u, double obs_v,
                   double fx, double fy, double cx, double cy,
                   double info_uv);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override;

    static ceres::CostFunction* Create(double obs_u, double obs_v,
                                        double fx, double fy, double cx, double cy,
                                        double info_uv) {
        return new MonoReprojCost(obs_u, obs_v, fx, fy, cx, cy, info_uv);
    }

private:
    double obs_u_, obs_v_;
    double fx_, fy_, cx_, cy_;
    double sqrt_info_uv_;
};

}  // namespace slam
