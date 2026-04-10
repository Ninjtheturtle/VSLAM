# Stereo Visual SLAM - from scratch

[![Watch the video](https://img.youtube.com/vi/4-7MSQ-har8/maxresdefault.jpg)](https://youtu.be/4-7MSQ-har8)

> KITTI sequence 00 - green is my estimate, orange is ground truth, white dots are the live map point cloud

---

## why i built this

I never really understood how an autonomous system could navigate a complex city without causing complete chaos. How does it weave through pedestrians or moving vehicles? I found the problem so puzzling that I wanted to take a stab at it without any outside help.

That meant NO pre-built perception stacks. NO black-box libraries. NO deep learning priors. Just raw pixels in, and a 6-DOF pose out.

How hard could it be?

Cut to four months later, and I've pretty much lost all my hair. Between late nights reading SLAM textbooks, studying branches of mathematics I didn't know existed, and experiencing integration hell for the tenth time in a single week, I finally have something to show for it.

---

## what it actually does

**stereo in, trajectory out.** left and right grayscale images come in, and the system:

1. initializes from a single stereo frame, triangulates 300-500 metric map points using `Z = fx * b / d`
2. tracks frame-to-frame using KLT optical flow with forward-backward consistency filtering
3. estimates pose each frame via motion-only bundle adjustment (Ceres, fixed 3D points, constant-velocity prediction)
4. inserts keyframes based on translation/rotation/track-coverage triggers
5. at each keyframe: extracts ORB features, stereo-matches grafted KLT tracks, triangulates new map points
6. runs sliding-window bundle adjustment (12 keyframes, Ceres SPARSE_SCHUR, analytical Jacobians, stereo + mono residuals)
7. streams everything to [Rerun](https://rerun.io/) for live 3D visualization

no scale ambiguity — stereo gives you real-world meters from frame one. the KITTI baseline is ~0.54m.

---

## the gpu side

**hamming matcher:** one block per query descriptor, 256 threads, warp-shuffle butterfly reduction. packs (distance, index) into a uint64 so the min reduction finds the argmin in one pass. three variants: raw nearest-neighbor, Lowe ratio test, and stereo epipolar (rejects matches that violate row alignment or disparity range before computing distance). runs asynchronously on a CUDA stream.

---

## bundle adjustment

the BA is the part i'm most proud of and the part that took the longest to get right.

**motion-only BA (per frame):** optimizes only the 6-DOF pose against fixed 3D map points. uses a constant-velocity prediction as the initial guess. runs in ~5ms with 800 tracks.

**local BA (per keyframe):** sliding-window optimizer over the last 12 keyframes using Ceres with SPARSE_SCHUR. optimizes both poses and 3D points jointly. the analytical Jacobians are hand-derived - the stereo cost function has 3 residuals (left u, left v, right u) and i work out the full chain rule through quaternion rotation, projection, and disparity. information-weighted: disparity residual is attenuated at depth (sigma_Z grows quadratic with Z).

what the BA does NOT have: any kind of pitch/roll constraint or pose prior. i tried both — they caused more problems than they solved. the BA runs unconstrained (except fixing the oldest keyframe for gauge freedom) and that turns out to be enough.

there is also no pose-graph optimization layer. an earlier version had one over the co-visibility graph, but the loop-detection path was feeding it `kf->T_cw * query->T_cw.inverse()` as the relative-pose "measurement" — that's the current estimate, not an independent observation, so PGO was minimizing estimates against themselves. it was removed until a real visual loop closure is wired in.

---

## the tracking pipeline

tracking uses KLT optical flow — no descriptor matching on non-keyframe frames:

1. pre-build image pyramids at frame creation (reused by KLT)
2. propagate flow tracks from previous frame via `calcOpticalFlowPyrLK`
3. forward-backward consistency filtering (reject tracks with >1px round-trip error)
4. motion-only Ceres BA on surviving 2D-3D correspondences (8 iterations, Huber loss)
5. constant-velocity sanity check (reject if delta > 0.3 rad or 3m)
6. reprojection cull at optimized pose (3px threshold)
7. spatial bucketing to cap at 800 tracks

keyframe insertion triggers: 5m translation, 6.9 degrees rotation, 30 frames max, track AABB collapse, or track starvation.

---

## the state machine

- **NOT_INITIALIZED** — waiting for a good stereo frame to bootstrap
- **OK** — normal KLT tracking + motion-only BA
- **LOST** — tracking failed; map resets and re-initializes from the next frame

when the map resets, archived keyframes are preserved so the trajectory visualization never disappears.

---

## performance

measured on KITTI sequence 00 (4541 frames, ~3.7km) on an RTX 3050 laptop, `--no-viz`:

- **9.0 FPS end-to-end** (4541 frames in 502.7s), 906 keyframes (1 per 5 frames)
- per-component avg / max: load 23.4 / 498.9 ms, track 30.7 / 395.3 ms, track(KF) 117.7 / 395.3 ms, BA 256.9 / 683.0 ms (n=906), viz 5.2 / 25.0 ms, **total 110.5 / 1021.1 ms**
- BA dominates keyframe-frame cost; non-KF track frames are ~30 ms
- **ATE RMSE 12.99 m** over the full 3.7 km trajectory, **RPE_t 3.84 %** over 100-frame segments (avg segment 76.4 m)
- ATE turn RMSE 13.13 m (1270 frames), ATE straight RMSE 12.94 m (3271 frames)
- max Y deviation from start 18.4 m, max yaw error −8.65°, final yaw error −1.40°
- 2 LOST transitions in 4541 frames (recovered both times via map reset — trajectory preserved through the archive)

stereo initialization gives metric scale from frame one — no scale drift from monocular bootstrapping. per-component benchmark timing is built into the main loop (printed every 500 frames + final report).

---

## building it

you need Windows, MSVC (VS 2022), CUDA 12.x, and vcpkg.

```bat
vcpkg install opencv4[core,features2d,calib3d,highgui] --triplet x64-windows
vcpkg install ceres[eigensparse,schur] --triplet x64-windows
vcpkg install eigen3 --triplet x64-windows

cmake -B build -DCMAKE_TOOLCHAIN_FILE=<your-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Rerun SDK (0.22.1) downloads automatically via FetchContent.

GPU target is compute capability 8.6 (Ampere). change `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt for your card.

---

## running it

```bat
:: download KITTI odometry: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
:: place under data/dataset/sequences/00/ (with image_0/, image_1/, calib.txt, times.txt)

build\vslam.exe --sequence data/dataset/sequences/00

:: benchmarking (no visualization overhead)
build\vslam.exe --sequence data/dataset/sequences/00 --no-viz
```

launch Rerun (`rerun`) before or alongside — the viewer connects on `127.0.0.1:9876`.

flags: `--start N`, `--end N`, `--no-viz`

---

## project structure

```
VSLAM/
├── src/                    tracker, BA, map, visualizer, main loop
├── include/slam/           pipeline headers
├── include/cuda/           CUDA kernel declarations
├── cuda/                   hamming matcher GPU kernel
├── data/dataset/           KITTI sequences + ground truth (not in git)
├── .vscode/                IDE configuration
├── CMakeLists.txt          build config (CUDA arch=86, vcpkg, FetchContent Rerun)
└── vcpkg.json              package manager dependencies
```

---
