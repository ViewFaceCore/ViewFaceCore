//
// Created by kier on 19-6-17.
//

#include "backend/base/base_activation.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "kernels/gpu/operator_on_gpu.h"
#include "kernels/common/math.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernels/gpu/gpu_kernel.h>
#include <kernels/gpu/cudax_fp16_math.h>
#include "global/fp16_operator_factory.h"

namespace ts {
    namespace gpu {
        template<typename T>
        __global__ static void abs_kernel(const T* input_data, T* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                auto val = input_data[index];
                output_data[index] = val > 0 ? val : -val;
            }
        }
        template<>
        __global__ void abs_kernel<float>(const float* input_data, float* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                auto val = input_data[index];
                output_data[index] = fabsf(val);
            }
        }
        template<>
        __global__ void abs_kernel<double>(const double* input_data, double* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                auto val = input_data[index];
                output_data[index] = fabs(val);
            }
        }

        template<typename T>
        void gpu_abs_compute_run(const Tensor &x, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();
            // int bytes_num = count * sizeof(T);

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            RUN_KERNEL(abs_kernel<T>, gridSize, blockSize, input_data, output_data, count);
        }

#ifdef TS_USE_CUDA_FP16
        __global__ void abs_kernel(const half* input_data, half* output_data, int size, half zero) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                auto val = input_data[index];
                output_data[index] = val > zero ? val : -val;
            }
        }

        template<>
        void gpu_abs_compute_run<half>(const Tensor &x, Tensor &out) {
            using T = half;
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();
            // int bytes_num = count * sizeof(T);

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            half zero = __float2half(0);
            RUN_KERNEL(abs_kernel, gridSize, blockSize, input_data, output_data, count, zero);
        }
#endif

        class Abs : public OperatorOnGPU<base::Activation> {
        public:
            void active(const Tensor &x, Tensor &out) final {

                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_abs_compute_run<TYPE>(x, out); break; }
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
TS_REGISTER_OPERATOR(Abs, GPU, "abs")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Abs, GPU, "abs")
#endif
