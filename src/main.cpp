// entry point , parses args, loads kitti seq, runs the main loop
// usage: vslam.exe --sequence <path> [--start N] [--end N] [--no-viz]

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/local_ba.hpp"
#include "slam/map.hpp"
#include "slam/tracker.hpp"
#include "slam/visualizer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// kitti sequence loader

struct KittiSequence {
    std::string sequence_path;
    std::vector<std::string> image_paths;        // left  (image_0/)
    std::vector<std::string> image_right_paths;  // right (image_1/); empty if not found
    std::vector<double> timestamps;
    slam::Camera camera;

    static KittiSequence load(const std::string& seq_path) {
        KittiSequence seq;
        seq.sequence_path = seq_path;

        seq.camera = slam::Camera::from_kitti_calib(seq_path + "/calib.txt");

        std::ifstream tf(seq_path + "/times.txt");
        if (!tf.is_open()) throw std::runtime_error("Cannot open times.txt in " + seq_path);
        double t;
        while (tf >> t) seq.timestamps.push_back(t);

        fs::path img_dir = fs::path(seq_path) / "image_0";
        if (!fs::exists(img_dir)) throw std::runtime_error("image_0/ not found in " + seq_path);

        std::vector<fs::path> paths;
        for (auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".png") paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        for (auto& p : paths) seq.image_paths.push_back(p.string());

        if (seq.image_paths.empty())
            throw std::runtime_error("No .png images found in " + img_dir.string());

        // optional right cam , enables stereo mode
        fs::path img_dir_r = fs::path(seq_path) / "image_1";
        if (fs::exists(img_dir_r)) {
            std::vector<fs::path> rpaths;
            for (auto& entry : fs::directory_iterator(img_dir_r)) {
                if (entry.path().extension() == ".png") rpaths.push_back(entry.path());
            }
            std::sort(rpaths.begin(), rpaths.end());
            for (auto& p : rpaths) seq.image_right_paths.push_back(p.string());
        }

        std::cout << "[KITTI] Loaded " << seq.image_paths.size() << " frames from " << seq_path;
        if (!seq.image_right_paths.empty())
            std::cout << " (stereo, b=" << seq.camera.baseline << " m)";
        std::cout << "\n";

        std::cout << "[KITTI] Intrinsics: fx=" << seq.camera.fx << "  fy=" << seq.camera.fy
                  << "  cx=" << seq.camera.cx << "  cy=" << seq.camera.cy << "\n";
        if (seq.camera.is_stereo()) {
            std::cout << "[KITTI] Stereo baseline: " << seq.camera.baseline << " m";
            if (seq.camera.baseline < 0.3 || seq.camera.baseline > 0.8)
                std::cout << "  *** WARNING: outside expected range [0.30, 0.80] m"
                             " — verify calib.txt uses P0/P1 (grayscale), not P2/P3 (color)";
            std::cout << "\n";
        }

        return seq;
    }
};

// args

struct Args {
    std::string sequence_path;
    int start_idx = 0;
    int end_idx = -1;  // -1 = all frames
    bool no_viz = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sequence" && i + 1 < argc) {
            args.sequence_path = argv[++i];
        } else if (a == "--start" && i + 1 < argc) {
            args.start_idx = std::stoi(argv[++i]);
        } else if (a == "--end" && i + 1 < argc) {
            args.end_idx = std::stoi(argv[++i]);
        } else if (a == "--no-viz") {
            args.no_viz = true;
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: vslam.exe --sequence <path> [--start N] [--end N] [--no-viz]\n";
            exit(0);
        }
    }
    if (args.sequence_path.empty()) {
        std::cerr << "Error: --sequence <path> is required\n";
        exit(1);
    }
    return args;
}

// hacky path math , assumes kitti dir layout: .../sequences/00 -> .../poses/00.txt
static std::string derive_gt_path(const std::string& seq_path) {
    std::string p = seq_path;
    while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    size_t s1 = p.find_last_of("/\\");
    std::string seq_id = (s1 == std::string::npos) ? p : p.substr(s1 + 1);
    std::string up1 = (s1 == std::string::npos) ? "." : p.substr(0, s1);
    size_t s2 = up1.find_last_of("/\\");
    std::string base = (s2 == std::string::npos) ? "." : up1.substr(0, s2);
    return base + "/poses/" + seq_id + ".txt";
}

// returns (tx, ty, tz) camera centers per frame , for viz & ATE
static std::vector<std::array<float, 3>> load_gt_centers(const std::string& path) {
    std::vector<std::array<float, 3>> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        double v[12];
        for (int i = 0; i < 12; ++i) ss >> v[i];
        out.push_back({(float)v[3], (float)v[7], (float)v[11]}); // indices 3,7,11 = tx,ty,tz in row-major 3x4
    }
    return out;
}

// full T_wc per frame , only needed for yaw diagnostics
static std::vector<Eigen::Isometry3d> load_gt_poses(const std::string& path) {
    std::vector<Eigen::Isometry3d> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        double v[12];
        for (int i = 0; i < 12; ++i) ss >> v[i];
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() << v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10];
        T.translation() << v[3], v[7], v[11];
        out.push_back(T);
    }
    return out;
}

// KITTI cam: X=right, Y=down, Z=fwd , yaw rotates around Y
static double extract_yaw(const Eigen::Matrix3d& R) { return std::atan2(R(0, 2), R(0, 0)); }

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    cv::setRNGSeed(42); // reproducible RANSAC results

    KittiSequence seq = KittiSequence::load(args.sequence_path);

    // hacky: calib.txt doesn't include image dims so we peek at the first frame
    {
        cv::Mat img = cv::imread(seq.image_paths[0], cv::IMREAD_GRAYSCALE);
        seq.camera.width = img.cols;
        seq.camera.height = img.rows;
    }

    // init
    auto map = slam::Map::create();
    auto local_ba = slam::LocalBA::create(seq.camera, map);
    slam::Visualizer::Ptr viz;

    auto tracker = slam::Tracker::create(seq.camera, map);

    std::string gt_path = derive_gt_path(args.sequence_path);

    if (!args.no_viz) {
        viz = slam::Visualizer::create();
        viz->log_pinhole(seq.camera);

        auto gt_centers_viz = load_gt_centers(gt_path);
        if (!gt_centers_viz.empty()) {
            viz->log_ground_truth(gt_centers_viz);
        }
    }

    int n_frames = static_cast<int>(seq.image_paths.size());
    int start_idx = std::max(0, args.start_idx);
    int end_idx = (args.end_idx < 0) ? n_frames : std::min(args.end_idx, n_frames);

    auto gt_centers_metrics = load_gt_centers(gt_path);
    auto gt_poses = load_gt_poses(gt_path);

    std::vector<std::array<float, 3>> est_centers(n_frames, {0.f, 0.f, 0.f});
    std::vector<bool> est_valid(n_frames, false);

    // per-frame metric accumulators
    double prev_yaw_gt = 0.0, prev_yaw_est = 0.0;
    bool prev_yaw_valid = false;
    double ate_turn_sum2 = 0.0, ate_straight_sum2 = 0.0;
    int ate_turn_n = 0, ate_straight_n = 0;
    double max_yaw_err = 0.0;
    double final_yaw_err = 0.0;
    int lost_count = 0;

    // main loop
    long frame_count = 0;
    auto t_start_wall = std::chrono::steady_clock::now();

    // timing
    struct BenchStats {
        double sum = 0, max = 0; int n = 0;
        void add(double v) { sum += v; if (v > max) max = v; ++n; }
        double avg() const { return n > 0 ? sum / n : 0; }
    };
    BenchStats bm_load, bm_track, bm_ba, bm_viz, bm_total;
    BenchStats bm_kf_track;  // tracking time on KF frames only

    for (int i = start_idx; i < end_idx; ++i) {
        auto tf_start = std::chrono::high_resolution_clock::now();

        cv::Mat img = cv::imread(seq.image_paths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "[VSLAM] Failed to load: " << seq.image_paths[i] << "\n";
            continue;
        }

        double ts = (i < (int)seq.timestamps.size()) ? seq.timestamps[i] : (double)i;
        auto frame = slam::Frame::create(img, ts, i);

        if (i < (int)seq.image_right_paths.size()) {
            frame->image_right = cv::imread(seq.image_right_paths[i], cv::IMREAD_GRAYSCALE);
        }

        auto t_load_end = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t_load_end - tf_start).count();
        bm_load.add(load_ms);

        // track
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = tracker->track(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        bm_track.add(track_ms);

        // local BA , KF frames only
        double ba_ms = 0.0;
        if (frame->is_keyframe && map->num_keyframes() >= 2) {
            local_ba->optimize();
            tracker->notify_ba_update();
            auto t2 = std::chrono::high_resolution_clock::now();
            ba_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            bm_ba.add(ba_ms);
            bm_kf_track.add(track_ms);
        }

        // viz
        auto tv0 = std::chrono::high_resolution_clock::now();
        if (viz) {
            viz->log_frame(frame);
            viz->log_trajectory(map, frame, ts);
            if (frame->is_keyframe) {
                viz->log_map(map, ts);
            }
        }
        auto tv1 = std::chrono::high_resolution_clock::now();
        double viz_ms = std::chrono::duration<double, std::milli>(tv1 - tv0).count();
        bm_viz.add(viz_ms);

        auto tf_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(tf_end - tf_start).count();
        bm_total.add(total_ms);

        ++frame_count;

        Eigen::Vector3d pos = frame->camera_center();
        est_centers[i] = {(float)pos.x(), (float)pos.y(), (float)pos.z()};
        est_valid[i] = true;

        // yaw + ATE vs ground truth
        double frame_ate = 0.0;
        if (i < (int)gt_poses.size()) {
            Eigen::Matrix3d R_wc_est = frame->T_wc().rotation();
            Eigen::Matrix3d R_wc_gt = gt_poses[i].rotation();
            double yaw_est = extract_yaw(R_wc_est) * 180.0 / 3.14159265358979323846;
            double yaw_gt = extract_yaw(R_wc_gt) * 180.0 / 3.14159265358979323846;
            double yaw_err = yaw_est - yaw_gt;
            while (yaw_err > 180.0) yaw_err -= 360.0;
            while (yaw_err < -180.0) yaw_err += 360.0;

            bool is_turn = false;
            if (prev_yaw_valid) {
                double yaw_rate_gt = yaw_gt - prev_yaw_gt;
                while (yaw_rate_gt > 180.0) yaw_rate_gt -= 360.0;
                while (yaw_rate_gt < -180.0) yaw_rate_gt += 360.0;
                is_turn = std::abs(yaw_rate_gt) > 0.5;
            }
            prev_yaw_gt = yaw_gt;
            prev_yaw_est = yaw_est;
            prev_yaw_valid = true;

            if (i < (int)gt_centers_metrics.size()) {
                double dx = pos.x() - gt_centers_metrics[i][0];
                double dy = pos.y() - gt_centers_metrics[i][1];
                double dz = pos.z() - gt_centers_metrics[i][2];
                frame_ate = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (is_turn) { ate_turn_sum2 += frame_ate * frame_ate; ++ate_turn_n; }
                else         { ate_straight_sum2 += frame_ate * frame_ate; ++ate_straight_n; }
            }

            if (std::abs(yaw_err) > std::abs(max_yaw_err)) max_yaw_err = yaw_err;
            final_yaw_err = yaw_err;
        }

        if (!ok) ++lost_count;

        // compact per-frame log: timing breakdown + position + ATE
        fprintf(stderr,
                "[%05d] load=%.0f trk=%.0f ba=%.0f viz=%.0f TOTAL=%.0fms | "
                "flow=%d kf=%zu pts=%zu ate=%.1f %s%s\n",
                i, load_ms, track_ms, ba_ms, viz_ms, total_ms,
                (int)frame->flow_pts.size(), map->num_keyframes(),
                map->num_map_points(), frame_ate,
                frame->is_keyframe ? "KF " : "",
                ok ? "OK" : "LOST");

        // benchmark checkpoint every 500 frames
        if (frame_count % 500 == 0) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start_wall).count();
            fprintf(stderr,
                    "\n=== BENCHMARK @ frame %d (%.1f FPS avg) ===\n"
                    "  load:  avg=%.1f  max=%.1f ms\n"
                    "  track: avg=%.1f  max=%.1f ms  (KF-track avg=%.1f)\n"
                    "  BA:    avg=%.1f  max=%.1f ms  (n=%d)\n"
                    "  viz:   avg=%.1f  max=%.1f ms\n"
                    "  TOTAL: avg=%.1f  max=%.1f ms\n"
                    "  KF rate: 1 per %.1f frames\n\n",
                    i, frame_count / elapsed,
                    bm_load.avg(), bm_load.max,
                    bm_track.avg(), bm_track.max, bm_kf_track.avg(),
                    bm_ba.avg(), bm_ba.max, bm_ba.n,
                    bm_viz.avg(), bm_viz.max,
                    bm_total.avg(), bm_total.max,
                    bm_ba.n > 0 ? (double)frame_count / bm_ba.n : 0.0);
        }
    }

    // summary
    auto t_end_wall = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end_wall - t_start_wall).count();
    double fps = frame_count / elapsed_s;

    std::cout << "\n[VSLAM] Done. " << frame_count << " frames in " << elapsed_s << "s = " << fps
              << " FPS\n"
              << "  Keyframes : " << map->num_keyframes() << "\n"
              << "  Map points: " << map->num_map_points() << "\n";

    // final benchmark
    fprintf(stderr,
            "\n========== FINAL BENCHMARK ==========\n"
            "Frames: %ld  |  FPS: %.1f  |  KFs: %d (1 per %.1f frames)\n"
            "Component      avg(ms)   max(ms)\n"
            "  load         %7.1f   %7.1f\n"
            "  track        %7.1f   %7.1f\n"
            "  track(KF)    %7.1f   %7.1f\n"
            "  BA           %7.1f   %7.1f  (n=%d)\n"
            "  viz          %7.1f   %7.1f\n"
            "  TOTAL        %7.1f   %7.1f\n"
            "TARGET: avg TOTAL <= 333ms (3 FPS)\n"
            "STATUS: %s\n"
            "======================================\n\n",
            frame_count, fps, bm_ba.n,
            bm_ba.n > 0 ? (double)frame_count / bm_ba.n : 0.0,
            bm_load.avg(), bm_load.max,
            bm_track.avg(), bm_track.max,
            bm_kf_track.avg(), bm_kf_track.max,
            bm_ba.avg(), bm_ba.max, bm_ba.n,
            bm_viz.avg(), bm_viz.max,
            bm_total.avg(), bm_total.max,
            bm_total.avg() <= 333.0 ? "PASS (>= 3 FPS)" : "FAIL (< 3 FPS)");

    // ATE + RPE
    if (!gt_centers_metrics.empty()) {
        double ate_sum2 = 0.0;
        double y_max_dev = 0.0;
        int ate_count = 0;
        double y_ref = gt_centers_metrics.empty() ? 0.0 : gt_centers_metrics[0][1];

        for (int i = start_idx; i < end_idx && i < (int)gt_centers_metrics.size(); ++i) {
            if (!est_valid[i]) continue;
            double dx = est_centers[i][0] - gt_centers_metrics[i][0];
            double dy = est_centers[i][1] - gt_centers_metrics[i][1];
            double dz = est_centers[i][2] - gt_centers_metrics[i][2];
            ate_sum2 += dx * dx + dy * dy + dz * dz;
            y_max_dev = std::max(y_max_dev, std::abs((double)est_centers[i][1] - y_ref));
            ++ate_count;
        }

        if (ate_count > 0) {
            double ate_rmse = std::sqrt(ate_sum2 / ate_count);
            std::cout << "\n[Metrics] ATE RMSE: " << ate_rmse << " m  (over " << ate_count
                      << " frames)\n";
            std::cout << "[Metrics] Max Y deviation from start: " << y_max_dev << " m\n";
        }

        // RPE over 100-frame windows (KITTI convention)
        const int rpe_delta = 100;
        double rpe_t_sum2 = 0.0;
        double rpe_seg_dist_sum = 0.0;
        int rpe_count = 0;

        for (int i = start_idx;
             i + rpe_delta < end_idx && i + rpe_delta < (int)gt_centers_metrics.size(); ++i) {
            if (!est_valid[i] || !est_valid[i + rpe_delta]) continue;

            double gt_dx = gt_centers_metrics[i + rpe_delta][0] - gt_centers_metrics[i][0];
            double gt_dy = gt_centers_metrics[i + rpe_delta][1] - gt_centers_metrics[i][1];
            double gt_dz = gt_centers_metrics[i + rpe_delta][2] - gt_centers_metrics[i][2];
            double gt_len = std::sqrt(gt_dx * gt_dx + gt_dy * gt_dy + gt_dz * gt_dz);
            if (gt_len < 1.0) continue; // skip near-stationary

            double e_dx = est_centers[i + rpe_delta][0] - est_centers[i][0];
            double e_dy = est_centers[i + rpe_delta][1] - est_centers[i][1];
            double e_dz = est_centers[i + rpe_delta][2] - est_centers[i][2];

            double err_dx = e_dx - gt_dx;
            double err_dy = e_dy - gt_dy;
            double err_dz = e_dz - gt_dz;
            double err = std::sqrt(err_dx * err_dx + err_dy * err_dy + err_dz * err_dz);
            rpe_t_sum2 += (err / gt_len) * (err / gt_len);
            rpe_seg_dist_sum += gt_len;
            ++rpe_count;
        }

        if (rpe_count > 0) {
            double rpe_t_pct = 100.0 * std::sqrt(rpe_t_sum2 / rpe_count);
            std::cout << "[Metrics] RPE_t: " << rpe_t_pct << "%  (over " << rpe_count
                      << " segments of " << rpe_delta << " frames, "
                      << "avg segment " << rpe_seg_dist_sum / rpe_count << " m)\n";
        }

        if (ate_turn_n > 0)
            std::cout << "[Metrics] ATE turn RMSE: " << std::sqrt(ate_turn_sum2 / ate_turn_n)
                      << " m  (" << ate_turn_n << " frames)\n";
        if (ate_straight_n > 0)
            std::cout << "[Metrics] ATE straight RMSE: "
                      << std::sqrt(ate_straight_sum2 / ate_straight_n) << " m  (" << ate_straight_n
                      << " frames)\n";
        std::cout << "[Metrics] Max yaw error: " << max_yaw_err << " deg\n";
        std::cout << "[Metrics] Final yaw error: " << final_yaw_err << " deg\n";
        std::cout << "[Metrics] LOST transitions: " << lost_count << "\n";
    }

    return 0;
}
