#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>      // for at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAException.h>     // for C10_CUDA_CHECK
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11)
#include <cuda_bf16.h>
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK(x.scalar_type() == dt, #x " has wrong dtype")
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template<typename T>
__device__ inline float to_float(T x);
template<>
__device__ inline float to_float<float>(float x){ return x; }
template<>
__device__ inline float to_float<half>(half x){ return __half2float(x); }
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11)
template<>
__device__ inline float to_float<nv_bfloat16>(nv_bfloat16 x){ return __bfloat162float(x); }
#endif

template<typename T>
__device__ inline T from_float(float x);
template<>
__device__ inline float from_float<float>(float x){ return x; }
template<>
__device__ inline half from_float<half>(float x){ return __float2half(x); }
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11)
template<>
__device__ inline nv_bfloat16 from_float<nv_bfloat16>(float x){ return __float2bfloat16(x); }
#endif

// blockDim.x threads; each thread handles one v-index within the tile.
// Shared mem layout: [q_f32[K]] [s_f32[BS]] [bp_f32[BS]] [red[blockDim.x]]
template<typename scalar_t>
__global__ void attn_fwd_q1_b1_stage1_kernel(
    const scalar_t* __restrict__ q,          // [HQ, K]
    const scalar_t* __restrict__ k,          // [HKV, T, K]
    const scalar_t* __restrict__ v,          // [T, HKV, V]
    float* __restrict__ m_buf,               // [HQ, NTB]
    float* __restrict__ l_buf,               // [HQ, NTB]
    float* __restrict__ o_buf,               // [HQ, NTB, V] (fp32)
    const float* __restrict__ qk_thresholds, // [HQ]
    // shapes/params
    int HQ, int HKV, int Kdim, int Vdim, int T,
    int G, int BS, int BV, int NTB,
    float scale, float rcp_ln2
){
    const int pid_v  = blockIdx.x;     // tile along V
    const int pid_hq = blockIdx.y;     // head q index
    const int pid_tb = blockIdx.z;     // time block index

    const int thread_id = threadIdx.x;
    const int nthreads  = blockDim.x;

    const int i_hq = pid_hq;
    const int i_h  = i_hq / G;

    const int s0   = pid_tb * BS;
    const int v0   = pid_v * BV;

    const int v_idx = v0 + thread_id;

    // shared memory
    extern __shared__ float smem[];
    float* sh_q  = smem;                 // [Kdim]
    float* sh_s  = sh_q + Kdim;          // [BS]
    float* sh_bp = sh_s + BS;            // [BS]
    float* sh_red= sh_bp + BS;           // [nthreads]
    __shared__ float sh_m_b;
    __shared__ float sh_l_b;
    __shared__ float sh_threshold;

    const float NEG_INF = -1e30f;
    const float scale2 = scale * rcp_ln2;

    // preload q into shared memory (fp32)
    const scalar_t* q_ptr = q + i_hq * Kdim;
    for (int kk = thread_id; kk < Kdim; kk += nthreads) {
        sh_q[kk] = to_float<scalar_t>(q_ptr[kk]);
    }
    if (thread_id == 0) {
        sh_threshold = qk_thresholds[i_hq];
    }
    __syncthreads();

    // compute s[t] for t in this block: s[t] = (q·k[i_h, t]) * scale2
    const scalar_t* k_head_base = k + i_h * (T * Kdim);
    for (int tloc = 0; tloc < BS; ++tloc) {
        const int tglob = s0 + tloc;
        float partial = 0.f;
        if (tglob < T) {
            const scalar_t* k_row = k_head_base + tglob * Kdim;
            for (int kk = thread_id; kk < Kdim; kk += nthreads) {
                float qv = sh_q[kk];
                float kv = to_float<scalar_t>(k_row[kk]);
                partial += qv * kv;
            }
        }
        // reduction across block
        sh_red[thread_id] = partial;
        __syncthreads();
        // parallel reduction to thread 0
        for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
            if (thread_id < offset) {
                sh_red[thread_id] += sh_red[thread_id + offset];
            }
            __syncthreads();
        }
        if (thread_id == 0) {
            float s_val = (tglob < T) ? sh_red[0] * scale2 : NEG_INF;
            sh_s[tloc] = s_val;
        }
        __syncthreads();
    }

    // compute m_b (max over active positions) and l_b, and store bp[t]
    if (thread_id == 0) {
        float m_b = NEG_INF;
        int num_active = 0;
        for (int tloc = 0; tloc < BS; ++tloc) {
            float s_val = sh_s[tloc];
            bool active = (s_val >= sh_threshold);
            if (active) {
                m_b = fmaxf(m_b, s_val);
                ++num_active;
            }
        }
        float l_b = 0.f;
        if (num_active > 0) {
            for (int tloc = 0; tloc < BS; ++tloc) {
                float s_val = sh_s[tloc];
                if (s_val >= sh_threshold) {
                    float p = exp2f(s_val - m_b);
                    sh_bp[tloc] = p;
                    l_b += p;
                } else {
                    sh_bp[tloc] = 0.f;
                }
            }
        } else {
            for (int tloc = 0; tloc < BS; ++tloc) sh_bp[tloc] = 0.f;
        }
        sh_m_b = m_b;
        sh_l_b = l_b;
    }
    __syncthreads();

    // compute o_b tile [BV] (each thread handles one v_idx)
    float ob = 0.f;
    const bool need_v = (sh_l_b > 0.f);
    if (need_v && v_idx < Vdim) {
        for (int tloc = 0; tloc < BS; ++tloc) {
            float p = sh_bp[tloc];
            if (p > 0.f) {
                int tglob = s0 + tloc;
                if (tglob < T) {
                    const scalar_t* v_row = v + (tglob * (HKV * Vdim)) + i_h * Vdim;
                    float vv = to_float<scalar_t>(v_row[v_idx]);
                    ob += p * vv;
                }
            }
        }
    }
    // write o_buf (always write; zeros if not needed)
    if (v_idx < Vdim) {
        float* o_tile = o_buf + ((i_hq * NTB + pid_tb) * Vdim) + v_idx;
        *o_tile = need_v ? ob : 0.f;
    }
    if (thread_id == 0) {
        m_buf[i_hq * NTB + pid_tb] = sh_m_b;
        l_buf[i_hq * NTB + pid_tb] = sh_l_b;
    }
}

// stage2: merge across tb, per (hq, v-tile)
template<typename out_t>
__global__ void attn_fwd_q1_b1_stage2_kernel(
    const float* __restrict__ m_buf,   // [HQ, NTB]
    const float* __restrict__ l_buf,   // [HQ, NTB]
    const float* __restrict__ o_buf,   // [HQ, NTB, V] (fp32)
    out_t* __restrict__ o,             // [HQ, V] (dtype of q)
    float* __restrict__ lse,           // [HQ] (fp32)
    int HQ, int Vdim, int NTB
){
    const int pid_v  = blockIdx.x; // tile along V
    const int pid_hq = blockIdx.y;
    const int thread_id = threadIdx.x;
    const int nthreads  = blockDim.x;

    const int v0 = pid_v * blockDim.x;
    const int v_idx = v0 + thread_id;

    float b_m = -1e30f;
    float b_acc = 0.f;
    float b_o   = 0.f;

    __shared__ float alpha, beta;
    __shared__ int has_blk;

    for (int tb = 0; tb < NTB; ++tb) {
        const float m_b = m_buf[pid_hq * NTB + tb];
        const float l_b = l_buf[pid_hq * NTB + tb];
        if (thread_id == 0) {
            const int has = (l_b > 0.f) ? 1 : 0;
            has_blk = has;
            // const float m_eff = has ? m_b : -CUDART_INF_F;
            const float m_eff = has ? m_b : -1e30f;
            const float new_m = fmaxf(b_m, m_eff);
            const float r_prev = exp2f(b_m - new_m);
            const float r_blk  = has ? exp2f(m_b - new_m) : 0.f;
            // update accumulators
            b_acc = b_acc * r_prev + l_b * r_blk;
            b_m   = new_m;
            alpha = r_prev;
            beta  = r_blk;
        }
        __syncthreads();

        // update numerator tile
        if (v_idx < Vdim) {
            float o_b = 0.f;
            if (has_blk) {
                o_b = o_buf[((pid_hq * NTB + tb) * Vdim) + v_idx];
            }
            b_o = b_o * alpha + o_b * beta;
        }
        __syncthreads();
    }

    if (v_idx < Vdim) {
        float out_val = b_o / b_acc;
        o[pid_hq * Vdim + v_idx] = from_float<out_t>(out_val);
    }
    if (blockIdx.x == 0 && thread_id == 0) {
        // base-2 lse
        lse[pid_hq] = b_m + log2f(b_acc);
    }
}

static inline int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// launcher
std::vector<at::Tensor> attn_fwd_q1_b1_splitT_cuda(
    at::Tensor q,            // [HQ, K]
    at::Tensor k,            // [HKV, T, K]
    at::Tensor v,            // [T, HKV, V]
    at::Tensor qk_thresholds,// [HQ], fp32
    double scale,
    int BS,
    int BV
){
    CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v); CHECK_CUDA(qk_thresholds);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v); CHECK_CONTIGUOUS(qk_thresholds);
    TORCH_CHECK(q.dim() == 2 && k.dim() == 3 && v.dim() == 3, "bad shapes");
    const int HQ   = q.size(0);
    const int Kdim = q.size(1);
    const int HKV  = k.size(0);
    const int T    = k.size(1);
    const int Kk   = k.size(2);
    TORCH_CHECK(Kk == Kdim, "K mismatch");
    const int Tv   = v.size(0);
    const int Hkv2 = v.size(1);
    const int Vdim = v.size(2);
    TORCH_CHECK(Tv == T && Hkv2 == HKV, "KV mismatch");
    TORCH_CHECK(HQ % HKV == 0, "GQA requires HQ % HKV == 0");
    const int G    = HQ / HKV;
    const int NTB  = CEIL_DIV(T, BS);
    const float RCP_LN2 = 1.4426950408889634f;

    auto device = q.device();
    auto dtype_q = q.scalar_type();
    auto o = at::empty({HQ, Vdim}, q.options());
    auto lse = at::empty({HQ}, q.options().dtype(at::kFloat));

    auto m_buf = at::empty({HQ, NTB}, q.options().dtype(at::kFloat));
    auto l_buf = at::empty({HQ, NTB}, q.options().dtype(at::kFloat));
    auto o_buf = at::empty({HQ, NTB, Vdim}, q.options().dtype(at::kFloat));

    dim3 grid1(CEIL_DIV(Vdim, BV), HQ, NTB);
    int threads = next_pow2(std::min(BV, 256));
    dim3 block1(threads);
    size_t smem_stage1 = (size_t)(Kdim + 2 * BS + threads) * sizeof(float);

    // dispatch by dtype of q/k/v/o
    if (dtype_q == at::kFloat) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            attn_fwd_q1_b1_stage1_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_stage1));
        attn_fwd_q1_b1_stage1_kernel<float><<<grid1, block1, smem_stage1, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            m_buf.data_ptr<float>(), l_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
            qk_thresholds.data_ptr<float>(),
            HQ, HKV, Kdim, Vdim, T, G, BS, BV, NTB, (float)scale, RCP_LN2);
    } else if (dtype_q == at::kHalf) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            attn_fwd_q1_b1_stage1_kernel<half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_stage1));
        attn_fwd_q1_b1_stage1_kernel<half><<<grid1, block1, smem_stage1, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            m_buf.data_ptr<float>(), l_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
            qk_thresholds.data_ptr<float>(),
            HQ, HKV, Kdim, Vdim, T, G, BS, BV, NTB, (float)scale, RCP_LN2);
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11)
    } else if (dtype_q == at::kBFloat16) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            attn_fwd_q1_b1_stage1_kernel<nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_stage1));
        attn_fwd_q1_b1_stage1_kernel<nv_bfloat16><<<grid1, block1, smem_stage1, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            m_buf.data_ptr<float>(), l_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
            qk_thresholds.data_ptr<float>(),
            HQ, HKV, Kdim, Vdim, T, G, BS, BV, NTB, (float)scale, RCP_LN2);
#endif
    } else {
        TORCH_CHECK(false, "Unsupported dtype for q/k/v");
    }

    // stage2
    dim3 grid2(CEIL_DIV(Vdim, threads), HQ);
    dim3 block2(threads);
    if (dtype_q == at::kFloat) {
        attn_fwd_q1_b1_stage2_kernel<float><<<grid2, block2, 0, at::cuda::getCurrentCUDAStream()>>>(
            m_buf.data_ptr<float>(), l_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
            o.data_ptr<float>(), lse.data_ptr<float>(),
            HQ, Vdim, NTB);
    } else if (dtype_q == at::kHalf) {
        attn_fwd_q1_b1_stage2_kernel<half><<<grid2, block2, 0, at::cuda::getCurrentCUDAStream()>>>(
            m_buf.data_ptr<float>(), l_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
            reinterpret_cast<half*>(o.data_ptr<at::Half>()), lse.data_ptr<float>(),
            HQ, Vdim, NTB);
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11)
    } else if (dtype_q == at::kBFloat16) {
        attn_fwd_q1_b1_stage2_kernel<nv_bfloat16><<<grid2, block2, 0, at::cuda::getCurrentCUDAStream()>>>(
            m_buf.data_ptr<float>(), l_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
            reinterpret_cast<nv_bfloat16*>(o.data_ptr<at::BFloat16>()), lse.data_ptr<float>(),
            HQ, Vdim, NTB);
#endif
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_fwd_q1_b1_splitT_cuda", &attn_fwd_q1_b1_splitT_cuda,
          "Attention qlen=1, split-T, 2-stage (CUDA)");
}