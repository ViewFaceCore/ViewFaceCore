#include <kernels/gpu/transpose.h>
#include <set>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>
#include <core/tensor_builder.h>


#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <kernels/gpu/gpu_kernel.h>


namespace ts {
    namespace gpu {
        template <typename T>
        static __global__ void gpu_transpose_kernel(int count, int *permute, const T*in, int *in_dim_weights, T *out, GpuHypeShape out_shape) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            int out_index = index;
            int in_index = 0;

            auto out_weight_it = out_shape.weights + 1;
            /* ============================================ */
            int running_dim = 0;
            /* -------------------------------------------- */

            for (int times = out_shape.dims - 1; times; --times) {
                auto coord = index / *out_weight_it;
                /* ============================================ */
                auto dim = permute[running_dim];
                in_index += coord * in_dim_weights[dim];
                ++running_dim;
                /* -------------------------------------------- */
                index %= *out_weight_it;
                ++out_weight_it;
            }
            auto coord = index;
            /* ============================================ */
            auto dim = permute[running_dim];
            in_index += coord * in_dim_weights[dim];
            /* -------------------------------------------- */

            /* ++++++++++++++++++++++++++++++++++++++++++++ */
            out[out_index] = in[in_index];
        }

        static inline Shape get_dim_weights(const Shape &shape) {
            if (shape.empty()) return Shape();
            Shape result(shape.size());
            result.back() = 1;
            auto out_it = result.rbegin() + 1;
            auto in_it = shape.rbegin();
            while (out_it != result.rend()) {
                *out_it = *in_it * *(out_it - 1);
                ++out_it;
                ++in_it;
            }
            return std::move(result);
        }

        template<typename T>
        static inline void gpu_transpose_compute_run(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            int *gpu_permute = nullptr;
            int *gpu_in_dim_weights = nullptr;

            T *out_data = out.data<T>();
            const T*in_data = x.data<T>();
            int count = out.count();

            Shape cpu_in_dim_weights = get_dim_weights(x.sizes());

            auto gpu_memory = MakeGPUHypeShape(out.device(), {out.sizes(), },
            {
                {(void*)(permute.data()), int(sizeof(int) * permute.size())},
                {(void*)(cpu_in_dim_weights.data()), int(sizeof(int) * cpu_in_dim_weights.size())},
            }, {
                (void **)(&gpu_permute),
                (void **)(&gpu_in_dim_weights),
            });
            auto &gpu_out_shape = gpu_memory.second[0];

            RUN_KERNEL(gpu_transpose_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, gpu_permute, in_data, gpu_in_dim_weights, out_data, gpu_out_shape);
        }

        void Transpose::transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_transpose_compute_run<TYPE>(x, permute, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT16, half);
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
TS_REGISTER_OPERATOR(Transpose, GPU, name::layer::transpose())
TS_REGISTER_FP16_OPERATOR(Transpose, GPU, name::layer::transpose())
