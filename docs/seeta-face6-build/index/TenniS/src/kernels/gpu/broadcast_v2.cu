#include <backend/base/base_broadcast_v2.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <kernels/gpu/operator_on_gpu.h>
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

#include <numeric>


namespace ts {
    namespace gpu {
        class BroadcastV2 : public OperatorOnGPU<base::BroadcastV2> {
        public:
            using self = BroadcastV2;
            using supper = OperatorOnGPU<base::BroadcastV2>;

            BroadcastV2() = default;

            void broadcast(const Tensor &x, Tensor &out) override;

            void broad_with_bias(const Tensor &x, Tensor &out, int dim) override;

            void broadcast_with_scalar(const Tensor &x, Tensor &out) override;

        };

        template<typename T>
        static __global__ void
        gpu_broadcast_kernel(int count, const T *C, T *out, GpuHypeShape C_shape, GpuHypeShape out_shape) {
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
        static inline void gpu_broadcast_compute_run(const Tensor &C, Tensor &out) {
            auto gpu_hype_shape = MakeGPUHypeShape(C.device(), {C.sizes(), out.sizes()});
            auto &C_hype_shape = gpu_hype_shape.second[0];
            auto &out_hype_shape = gpu_hype_shape.second[1];
            auto count = out.count();

            RUN_KERNEL(gpu_broadcast_kernel<T>,
                       CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, C.data<T>(), out.data<T>(), C_hype_shape, out_hype_shape);
        }

        template<typename T>
        static __global__ void gpu_broadcast_with_scalar_kernel(int N, T *dst, T val) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= N) return;
            dst[index] = val;
        }

        template<typename T>
        static inline void gpu_broadcast_with_scalar(const Tensor &x, Tensor &out) {
            auto val = x.data<T>()[0];
            auto pout = out.data<T>();
            auto count = out.count();

            RUN_KERNEL(gpu_broadcast_with_scalar_kernel<T>,
                       CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, pout, val);
        }

        template<typename T>
        static __global__ void gpu_broadcast_with_bias_kernel(int N, T *dst, int channels, int count, const T *px) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= N) return;
            auto c = index / count % channels;
            dst[index] = px[c];
        }

        template<typename T>
        static inline void gpu_broadcast_with_bias(const Tensor &x, Tensor &out, int dim) {
            auto px = x.data<T>();
            auto pout = out.data<T>();

            auto &out_shape = out.sizes();

            // auto number = std::accumulate(out_shape.begin(), out_shape.begin() + dim, 1, std::multiplies<int>());
            auto count = std::accumulate(out_shape.begin() + dim + 1, out_shape.end(), 1, std::multiplies<int>());

            auto channels = out_shape[dim];

            auto N = out.count();

            RUN_KERNEL(gpu_broadcast_with_bias_kernel<T>,
                       CUDA_BLOCK(N, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       N, pout, channels, count, px);
        }

        void BroadcastV2::broadcast(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { gpu_broadcast_compute_run<TYPE>(x, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void BroadcastV2::broad_with_bias(const Tensor &x, Tensor &out, int dim) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { gpu_broadcast_with_bias<TYPE>(x, out, dim); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void BroadcastV2::broadcast_with_scalar(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { gpu_broadcast_with_scalar<TYPE>(x, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(BroadcastV2, GPU, "broadcast")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(BroadcastV2, GPU, "broadcast")
#endif