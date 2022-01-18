//
// Created by kier on 19-3-31.
//

#include "kernels/gpu/gpu_gemm.h"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "kernels/gpu/cuda_context.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"

#ifdef TS_USE_CUBLAS
#include "kernels/gpu/math_cublas.h"
#else
#include "kernels/gpu/math_gpu.h"
#endif

#include "kernels/gpu/gpu_kernel.h"

#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"
#include "backend/name.h"

namespace ts {
    namespace gpu {
        template<typename T>
        static __global__ void gpu_gemm_broadcast_kernel(int count, const T*C, T*out, GpuHypeShape C_shape, GpuHypeShape out_shape) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            int out_index = index;
            int in_index = 0;

            auto out_weight_it = out_shape.weights + 1;
            auto in_weight_it = C_shape.weights + 1;
            /* ============================================ */
            auto in_shape_it = C_shape.shape;
            /* ============================================ */

            for (int times = out_shape.dims - 1; times; --times) {
                auto coord = index / *out_weight_it;
                /* ============================================ */
                coord %= *in_shape_it;
                ++in_shape_it;
                /* ============================================ */
                in_index += coord * *in_weight_it;
                index %= *out_weight_it;
                ++out_weight_it;
                ++in_weight_it;
            }
            auto coord = index;
            /* ============================================ */
            coord %= *in_shape_it;
            /* ============================================ */
            in_index += coord;

            /* ++++++++++++++++++++++++++++++++++++++++++++ */
            out[out_index] = C[in_index];
        }

        template<typename T>
        static inline void gpu_gemm_broadcast_compute_run(const Tensor &C, Tensor &out) {
            auto gpu_hype_shape = MakeGPUHypeShape(C.device(), {C.sizes(), out.sizes()});
            auto &C_hype_shape = gpu_hype_shape.second[0];
            auto &out_hype_shape = gpu_hype_shape.second[1];
            auto count = out.count();

            RUN_KERNEL(gpu_gemm_broadcast_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, C.data<T>(), out.data<T>(), C_hype_shape, out_hype_shape);
        }

        template<typename T>
        static void gpu_gemm_compute_run(const Tensor &A, const Tensor &B, const Tensor &C, int K,
                                  float alpha, float beta, bool transA, bool transB, Tensor &out) {

            auto ptr_A = A.data<T>();
            auto ptr_B = B.data<T>();
            auto ptr_C = out.data<T>();

            int M = out.size(0);
            int N = out.size(1);

            // broadcast C to output
            if (!near_zero(beta)) {
                if (C.has_shape(out.sizes())) {
                    auto dst = out.weak_memory();
                    auto src = C.weak_memory();
                    memcpy(dst, src);
                } else {
                    gpu_gemm_broadcast_compute_run<T>(C, out);
                }
            } else {
                beta = 0;
            }
#ifdef TS_USE_CUBLAS
            auto &context = ctx::ref<DeviceContext>();
            CUDAContextHandle *handle = reinterpret_cast<CUDAContextHandle *>(context.handle);
            auto cublas_handle = handle->cublas_handle();

            auto cublas_transA = transA ? cublas::Trans : cublas::NoTrans;
            auto cublas_transB = transB ? cublas::Trans : cublas::NoTrans;

            cublas::math<T>::gemm(cublas_handle,
                    cublas_transA, cublas_transB, M, N, K,
                    (T) alpha, ptr_A, ptr_B, (T) beta, ptr_C);
#else
            auto cublas_transA = transA ? cublas::Trans : cublas::NoTrans;
            auto cublas_transB = transB ? cublas::Trans : cublas::NoTrans;
            gpu::math<T>::gemm(
                    cublas_transA, cublas_transB, M, N, K,
                    (T) alpha, ptr_A, ptr_B, (T) beta, ptr_C);
#endif
        }

        void Gemm::gemm(const Tensor &A, const Tensor &B, const Tensor &C, int K,
                        float alpha, float beta, bool transA, bool transB, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_gemm_compute_run<TYPE>(A, B, C, K, alpha, beta, transA, transB, out); break; }
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Gemm, GPU, name::layer::gemm())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Gemm, GPU, name::layer::gemm())
#endif