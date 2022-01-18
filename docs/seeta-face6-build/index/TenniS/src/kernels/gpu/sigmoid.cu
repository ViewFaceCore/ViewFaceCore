#include <kernels/gpu/sigmoid.h>

#include "backend/name.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/gpu_kernel.h"

#include "kernels/gpu/cudax_fp16_math.h"

namespace ts {
    namespace gpu {
        template<typename T>
        __global__ static void sigmoid_kernel(const T* input_data, T* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                output_data[index] = T(1) / (T(1) + exp(-input_data[index]));
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__ void sigmoid_kernel<half>(const half* input_data, half* output_data, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            half one(1.f);
            if (index < size)
            {
                output_data[index] = one / (one + half(exp(-input_data[index])));
            }
        }
#endif

        template<typename T>
        static void cpu_sigmoid_compute_run(const Tensor &x, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            // int bytes_num = count * sizeof(T);

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            RUN_KERNEL(sigmoid_kernel<T>, gridSize, blockSize, input_data, output_data, count);
        }

        void Sigmoid::active(const Tensor &x, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_sigmoid_compute_run<TYPE>(x, out); break; }
                //DECLARE_COMPUTE_RUN(INT8, int8_t);
                //DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                //DECLARE_COMPUTE_RUN(INT16, int16_t);
                //DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                //DECLARE_COMPUTE_RUN(INT32, int32_t);
                //DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                //DECLARE_COMPUTE_RUN(INT64, int64_t);
                //DECLARE_COMPUTE_RUN(UINT64, uint64_t);
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
TS_REGISTER_OPERATOR(Sigmoid, ts::GPU, name::layer::sigmoid())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Sigmoid, ts::GPU, name::layer::sigmoid())
#endif
