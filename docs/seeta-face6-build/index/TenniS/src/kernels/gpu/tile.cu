//
// Created by kier on 19-7-23.
//

#include "backend/base/base_tile.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "kernels/gpu/operator_on_gpu.h"
#include "kernels/common/math.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernels/gpu/gpu_kernel.h>
#include <kernels/gpu/cudax_fp16_math.h>
#include "global/fp16_operator_factory.h"

#include "backend/name.h"

namespace ts {
    namespace gpu {
        template <typename T>
        static __global__ void tile_gpu_kernel(int count, const T *in, T *out, GpuHypeShape in_shape, GpuHypeShape out_shape) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            int out_index = index;
            int in_index = 0;

            auto out_weight_it = out_shape.weights + 1;
            auto in_weight_it = in_shape.weights + 1;
            /* ============================================ */
            auto in_shape_it = in_shape.shape;
            /* -------------------------------------------- */

            for (int times = out_shape.dims - 1; times; --times) {
                auto coord = index / *out_weight_it;
                /* ============================================ */
                coord %= *in_shape_it;
                ++in_shape_it;
                /* -------------------------------------------- */
                in_index += coord * *in_weight_it;
                index %= *out_weight_it;
                ++out_weight_it;
                ++in_weight_it;
            }
            auto coord = index;
            /* ============================================ */
            coord %= *in_shape_it;
            /* -------------------------------------------- */
            in_index += coord;

            /* ++++++++++++++++++++++++++++++++++++++++++++ */
            out[out_index] = in[in_index];
        }

        template <typename T>
        static inline void gpu_tile_compute_run(const Tensor &x, const std::vector<int32_t> &repeats, Tensor &out) {
            auto gpu_memory = MakeGPUHypeShape(out.device(), {x.sizes(), out.sizes()});
            auto &gpu_in_shape = gpu_memory.second[0];
            auto &gpu_out_shape = gpu_memory.second[1];
            auto in_data = x.data<T>();
            auto out_data = out.data<T>();
            auto count = out.count();

            RUN_KERNEL(tile_gpu_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, in_data, out_data, gpu_in_shape, gpu_out_shape);
        }


        class Tile : public OperatorOnGPU<base::Tile> {
        public:
            void tile(const Tensor &x, const std::vector<int32_t> &repeats, Tensor &out) final {

                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_tile_compute_run<TYPE>(x, repeats, out); break; }
                    DECLARE_COMPUTE_RUN(INT8, int8_t);
                    DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                    DECLARE_COMPUTE_RUN(INT16, int16_t);
                    DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                    DECLARE_COMPUTE_RUN(INT32, int32_t);
                    DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                    DECLARE_COMPUTE_RUN(INT64, int64_t);
                    DECLARE_COMPUTE_RUN(UINT64, uint64_t);
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
        };
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Tile, GPU, name::layer::tile())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Tile, ts::GPU, name::layer::tile())
#endif