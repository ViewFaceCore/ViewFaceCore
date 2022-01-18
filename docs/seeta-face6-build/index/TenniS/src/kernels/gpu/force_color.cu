//
// Created by kier on 19-6-27.
//

#include "backend/base/base_force_color.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "kernels/gpu/operator_on_gpu.h"
#include "kernels/common/math.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernels/gpu/gpu_kernel.h>
#include "backend/name.h"
#include <numeric>

namespace ts {
    namespace gpu {
        template<typename T>
        __global__ static void force_color_kernel(const T* input_data, T* output_data, int size, int input_channels, int output_channels) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < size)
            {
                auto input_pixel = &input_data[i * input_channels];
                auto output_pixel = &output_data[i * output_channels];
                for (int j = 0; j < output_channels; ++j) {
                    output_pixel[j] = input_pixel[0];
                }
            }
        }

        template<typename T>
        void gpu_force_color_compute_run(const Tensor &x, Tensor &out) {
            auto &size = x.sizes();
            auto dims = x.dims();
            auto number = std::accumulate(size.begin(), size.end() - 1, 1, std::multiplies<int32_t>());
            auto input_channels = x.size(dims - 1);
            auto output_channels = out.size(dims - 1);

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(number, blockSize.x));

            RUN_KERNEL(force_color_kernel<T>, gridSize, blockSize,
                       input_data, output_data, number, input_channels, output_channels);
        }

        class ForceColor : public OperatorOnGPU<base::ForceColor> {
        public:
            void force_color(const Tensor &x, Tensor &out) final {
                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_force_color_compute_run<TYPE>(x, out); break; }
                    DECLARE_COMPUTE_RUN(INT8, int8_t);
                    DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                    DECLARE_COMPUTE_RUN(INT16, int16_t);
                    DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                    DECLARE_COMPUTE_RUN(INT32, int32_t);
                    DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                    DECLARE_COMPUTE_RUN(INT64, int64_t);
                    DECLARE_COMPUTE_RUN(UINT64, uint64_t);
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
TS_REGISTER_OPERATOR(ForceColor, GPU, name::layer::force_color())