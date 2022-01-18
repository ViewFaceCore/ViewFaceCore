#include <backend/base/base_broadcast.h>

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


namespace ts {
    namespace gpu {
        class Broadcast : public OperatorOnGPU<base::Broadcast> {
        public:
            using self = Broadcast;
            using supper = OperatorOnGPU<base::Broadcast>;

            Broadcast() = default;

            void broadcast(const Tensor &x, const std::vector<int32_t> &shape, Tensor &out) override;
        };

        template<typename T>
        static __global__ void gpu_broadcast_kernel(int count, const T*C, T*out, GpuHypeShape C_shape, GpuHypeShape out_shape) {
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

            RUN_KERNEL(gpu_broadcast_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, C.data<T>(), out.data<T>(), C_hype_shape, out_hype_shape);
        }


        void Broadcast::broadcast(const Tensor &x, const std::vector<int32_t> &shape, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_broadcast_compute_run<TYPE>(x, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, int8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, int16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, int32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, int64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, int32_t);
                DECLARE_COMPUTE_RUN(FLOAT64, int64_t);
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, int16_t);
#endif
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
// TS_REGISTER_OPERATOR(Broadcast, GPU, "broadcast")
#ifdef TS_USE_CUDA_FP16
// TS_REGISTER_FP16_OPERATOR(Broadcast, GPU, "broadcast")
#endif
