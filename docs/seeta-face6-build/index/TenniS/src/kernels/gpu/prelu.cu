#include <kernels/gpu/prelu.h>
#include <core/tensor_builder.h>
#include "backend/name.h"

#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {
        template<typename T>
        __global__ static void prelu_kernel(const T* input_data, T* output_data,const T*slope, int dim_num, int last_dims, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                int slope_index = index % (dim_num * last_dims) / last_dims;
                T val = input_data[index];
                T max_temp = val > T(0) ? val : T(0);
                T min_temp = val < T(0) ? val : T(0);
                output_data[index] = max_temp + slope[slope_index] * min_temp;
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__  void prelu_kernel<half>(const half* input_data, half* output_data, const half*slope, int dim_num, int last_dims, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            half zero(0.f);
            if (index < size)
            {
                int slope_index = index % (dim_num * last_dims) / last_dims;
                half val = input_data[index];
                half max_temp = val > zero ? val : zero;
                half min_temp = val < zero ? val : zero;
                output_data[index] = max_temp + slope[slope_index] * min_temp;
            }
        }
#endif

        template <typename T>
        void cpu_prelu_compute_run(const Tensor &x, const Tensor &slope, int dim, Tensor &out) {
            auto output_shape = out.sizes();
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            const T *slope_data = slope.data<T>();

            int count = out.count();

            int dim_num = output_shape[dim];

            int last_dims = 1;
            for (int i = dim + 1; i < output_shape.size(); i++) {
                last_dims *= output_shape[i];
            }

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(count, blockSize.x));

            RUN_KERNEL(prelu_kernel<T>, gridSize, blockSize,
                       input_data, output_data, slope_data, dim_num, last_dims, count);
        }

        void PReLU::prelu(const Tensor &x, const Tensor &slope, int dim, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_prelu_compute_run<TYPE>(x, slope, dim, out); break; }
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
TS_REGISTER_OPERATOR(PReLU, GPU, name::layer::prelu())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(PReLU, GPU, name::layer::prelu())
#endif
