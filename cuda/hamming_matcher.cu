// brute-force ORB hamming matcher
// one block per query, threads stripe over train descs then warp-reduce to best

#include "cuda/hamming_matcher.cuh"
#include <cstring>
#include <limits>
#include <vector>

static constexpr int BLOCK_SIZE = 256;

// shm layout: [0..7] = query desc, rest = per-thread reduction scratch
// packing: (distance << 32 | index), uint64 min gives correct argmin

__global__ void hamming_match_kernel(
    const uint32_t* __restrict__ d_query,
    const uint32_t* __restrict__ d_train,
    int              N_t,
    int*  __restrict__ d_best_idx,
    int*  __restrict__ d_best_dist
)
{
    extern __shared__ uint32_t shm[];  // shm[0..7] = query desc

    const int qid = blockIdx.x;
    const int tid = threadIdx.x;

    // first 8 threads load the query descriptor
    if (tid < kDescUint32) {
        shm[tid] = d_query[qid * kDescUint32 + tid];
    }
    __syncthreads();

    // each thread scans train descs in strides of BLOCK_SIZE
    int   local_min_dist = kMaxHamming + 1;
    int   local_min_idx  = -1;

    for (int t = tid; t < N_t; t += BLOCK_SIZE) {
        const uint32_t* tptr = d_train + t * kDescUint32;

        // hamming = sum(popcount(q XOR t)) over 8 words
        int dist = 0;
        #pragma unroll
        for (int k = 0; k < kDescUint32; ++k) {
            dist += __popc(shm[k] ^ tptr[k]);
        }

        if (dist < local_min_dist) {
            local_min_dist = dist;
            local_min_idx  = t;
        }
    }

    // warp reduction, pack (dist, idx) into uint64 for single min pass
    uint64_t val = ((uint64_t)(uint32_t)local_min_dist << 32)
                 | ((uint64_t)(uint32_t)local_min_idx);

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        uint64_t other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (other < val) val = other;
    }

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    __shared__ uint64_t warp_vals[BLOCK_SIZE / 32];

    if (lane_id == 0) {
        warp_vals[warp_id] = val;
    }
    __syncthreads();

    // thread 0: final reduction across warps
    if (tid == 0) {
        const int n_warps = BLOCK_SIZE / 32;
        uint64_t best = warp_vals[0];
        for (int w = 1; w < n_warps; ++w) {
            if (warp_vals[w] < best) best = warp_vals[w];
        }
        d_best_dist[qid] = (int)(best >> 32);
        d_best_idx [qid] = (int)(best & 0xFFFFFFFFu);
    }
}

// two-NN variant for lowe ratio test

__global__ void hamming_match_ratio_kernel(
    const uint32_t* __restrict__ d_query,
    const uint32_t* __restrict__ d_train,
    int              N_t,
    float            ratio,
    int*  __restrict__ d_best_idx,
    int*  __restrict__ d_best_dist
)
{
    extern __shared__ uint32_t shm[];

    const int qid = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < kDescUint32) {
        shm[tid] = d_query[qid * kDescUint32 + tid];
    }
    __syncthreads();

    int local_d1 = kMaxHamming + 1, local_i1 = -1;
    int local_d2 = kMaxHamming + 1;

    for (int t = tid; t < N_t; t += BLOCK_SIZE) {
        const uint32_t* tptr = d_train + t * kDescUint32;
        int dist = 0;
        #pragma unroll
        for (int k = 0; k < kDescUint32; ++k) {
            dist += __popc(shm[k] ^ tptr[k]);
        }
        if (dist < local_d1) {
            local_d2 = local_d1;
            local_d1 = dist; local_i1 = t;
        } else if (dist < local_d2) {
            local_d2 = dist;
        }
    }

    // warp reduction tracking both best & second-best
    __shared__ int  warp_d1[BLOCK_SIZE / 32];
    __shared__ int  warp_i1[BLOCK_SIZE / 32];
    __shared__ int  warp_d2[BLOCK_SIZE / 32];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        int od1 = __shfl_down_sync(0xFFFFFFFF, local_d1, offset);
        int oi1 = __shfl_down_sync(0xFFFFFFFF, local_i1, offset);
        int od2 = __shfl_down_sync(0xFFFFFFFF, local_d2, offset);
        if (od1 < local_d1) {
            if (local_d1 < local_d2) local_d2 = local_d1;  // demote our best to second
            local_d1 = od1; local_i1 = oi1;
        } else {
            if (od1 < local_d2) local_d2 = od1;
        }
        if (od2 < local_d2) local_d2 = od2;
    }

    if (lane_id == 0) {
        warp_d1[warp_id] = local_d1;
        warp_i1[warp_id] = local_i1;
        warp_d2[warp_id] = local_d2;
    }
    __syncthreads();

    if (tid == 0) {
        const int n_warps = BLOCK_SIZE / 32;
        int best_d1 = warp_d1[0], best_i1 = warp_i1[0], best_d2 = warp_d2[0];
        for (int w = 1; w < n_warps; ++w) {
            int wd1 = warp_d1[w], wi1 = warp_i1[w], wd2 = warp_d2[w];
            if (wd1 < best_d1) {
                if (best_d1 < best_d2) best_d2 = best_d1;
                best_d1 = wd1; best_i1 = wi1;
            } else {
                if (wd1 < best_d2) best_d2 = wd1;
            }
            if (wd2 < best_d2) best_d2 = wd2;
        }

        bool accepted = (best_d1 < ratio * best_d2) && (best_i1 >= 0);
        d_best_idx [qid] = accepted ? best_i1 : -1;
        d_best_dist[qid] = accepted ? best_d1 : kMaxHamming;
    }
}

// host wrapper

void cuda_match_hamming(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    int*           h_best_idx,
    int*           h_best_dist
)
{
    if (N_q == 0 || N_t == 0) return;

    const size_t q_bytes = (size_t)N_q * kDescBytes;
    const size_t t_bytes = (size_t)N_t * kDescBytes;

    uint32_t *d_query = nullptr, *d_train = nullptr;
    int      *d_idx   = nullptr, *d_dist  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_query, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_train, t_bytes));
    CUDA_CHECK(cudaMalloc(&d_idx,   sizeof(int) * N_q));
    CUDA_CHECK(cudaMalloc(&d_dist,  sizeof(int) * N_q));

    CUDA_CHECK(cudaMemcpy(d_query, h_query, q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train, h_train, t_bytes, cudaMemcpyHostToDevice));

    const size_t shm_bytes = kDescUint32 * sizeof(uint32_t);  // 32 bytes for query desc

    dim3 grid(N_q, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hamming_match_kernel<<<grid, block, shm_bytes>>>(
        d_query, d_train, N_t, d_idx, d_dist
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_best_idx,  d_idx,  sizeof(int) * N_q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_dist, d_dist, sizeof(int) * N_q, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_train));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_dist));
}

