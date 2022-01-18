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
        __device__ static T sigmoid(T x) {
            return T(T(1) / (T(1) + exp(-x)));
        }

        template<typename T>
        __device__ static T tanh(T x) {
            return T(2) * sigmoid(T(2) * x) - T(1);
        }

        template<typename T>
        __global__ static void tanh_kernel(const T* input_data, T* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                T val = input_data[index];
                output_data[index] = tanh(val);
            }
        }

        template<typename T>
        void gpu_tanh_compute_run(const Tensor &x, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();
            // int bytes_num = count * sizeof(T);

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            RUN_KERNEL(tanh_kernel<T>, gridSize, blockSize, input_data, output_data, count);
        }

#ifdef TS_USE_CUDA_FP16
        __global__ void tanh_kernel(const half* input_data, half* output_data, int size, half one, half two) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                auto val = input_data[index];
                auto sigmoid = one / (one + exp(-(two * val)));
                output_data[index] = two * sigmoid - one;
            }
        }

        template <>
        void gpu_tanh_compute_run<half>(const Tensor &x, Tensor &out) {
            const half *input_data = x.data<half>();
            half *output_data = out.data<half>();
            int count = out.count();
            // int bytes_num = count * sizeof(T);

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            auto one = __float2half(1);
            auto two = __float2half(2);

            RUN_KERNEL(tanh_kernel, gridSize, blockSize, input_data, output_data, count, one, two);
        }
#endif

        class Tanh : public OperatorOnGPU<base::Activation> {
        public:
            void active(const Tensor &x, Tensor &out) final {

                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_tanh_compute_run<TYPE>(x, out); break; }
                    // DECLARE_COMPUTE_RUN(INT8, int8_t);
                    // DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                    // DECLARE_COMPUTE_RUN(INT16, int16_t);
                    // DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                    // DECLARE_COMPUTE_RUN(INT32, int32_t);
                    // DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                    // DECLARE_COMPUTE_RUN(INT64, int64_t);
                    // DECLARE_COMPUTE_RUN(UINT64, uint64_t);
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
TS_REGISTER_OPERATOR(Tanh, GPU, "tanh")
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Tanh, GPU, "tanh")
#endif
