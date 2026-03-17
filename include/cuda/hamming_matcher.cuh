#pragma once

#include <cstdint>
#include <cstdio>

// abort on CUDA failure w/ file:line
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA] %s:%d %s\n",                               \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ORB descriptors: 256 bits = 32 bytes = 8 x uint32
static constexpr int kDescBytes  = 32;
static constexpr int kDescUint32 = 8;
static constexpr int kMaxHamming = 256;

// GPU NN matcher for ORB hamming distance
// all ptrs are host, device alloc is internal

// best match idx + distance per query row
void cuda_match_hamming(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    int*           h_best_idx,
    int*           h_best_dist
);

// same but w/ lowe ratio test, retains only if best/second < ratio
void cuda_match_hamming_ratio(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    float          ratio,
    int*           h_best_idx,
    int*           h_best_dist
);

// stereo epipolar matcher: restricts candidates by
//   |y_q - y_t| <= epi_tol  and  d_min <= x_q - x_t <= d_max
// then lowe ratio test, writes best right idx per left desc
void cuda_match_stereo_epipolar(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    const float*   h_y_query,
    const float*   h_y_train,
    const float*   h_x_query,
    const float*   h_x_train,
    float          epi_tol,
    float          d_min,
    float          d_max,
    float          ratio,
    int*           h_best_idx,
    int*           h_best_dist
);
