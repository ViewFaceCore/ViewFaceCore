#include <kernels/gpu/batch_norm.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include "global/fp16_operator_factory.h"
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>

#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <runtime/runtime.h>

#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {
        template<typename T>
        static __global__ void gpu_batch_norm_compute_kernel(const T* data, T* out, int size, int step, int slice,
                                        const T* mean, const T* variance ) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                int dim = index % ( step * slice ) / (step);
                out[index] = (data[index] - mean[dim]) * variance[dim];
            }
        }

        template<typename T>
        static __global__ void vec_kernel(const int N, T one, T epsilon, const T* input,T* output) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            for (; index < N; index += blockDim.x * gridDim.x) {
                output[index] = one / sqrt(input[index] + epsilon);
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__ void vec_kernel<half>(const int N, half one, half epsilon, const half* input, half* output) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            for (; index < N; index += blockDim.x * gridDim.x) {
                output[index] = one / hsqrt(input[index] + epsilon);
            }
        }
#endif

        template<typename T>
        static void gpu_batch_norm_compute_run(const Tensor &x, const Tensor &mean,
                                               const Tensor &variance, int dim, float epsilon, Tensor &out) {
            const Shape &shape = x.sizes();
            //int predims = 1;
            int backdims = 1;
            //for (int i = 0; i < dim; i++) {
            //    predims *= shape[i];
            //}

            for (int i = dim + 1; i < shape.size(); i++) {
                backdims *= shape[i];
            }

            const T *psrc = x.data<T>();
            const T *pmean = mean.data<T>();
            const T *pvariance = variance.data<T>();
            T *pdst = out.data<T>();

            Shape vec_shape = variance.sizes();
            Tensor vec_tensor(RuntimeContext::FlowMemory(), variance.dtype(), vec_shape, MemoryDevice(variance.device()));
            T* vec_data = vec_tensor.data<T>();

            int vec_len = vec_tensor.count();
            dim3 block_size(CUDA_THREAD_NUM);
            dim3 grid_size(CUDA_BLOCK(vec_len, block_size.x));
            T one(1.f);
            T temp_epsilon(epsilon);
            RUN_KERNEL(vec_kernel<T>, grid_size, block_size, vec_len, one, temp_epsilon, pvariance,vec_data);

            RUN_KERNEL(gpu_batch_norm_compute_kernel<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       psrc, pdst, out.count(), backdims, shape[dim], pmean, vec_data);
        }


        void BatchNorm::batch_norm(const Tensor &x, const Tensor &mean, const Tensor &variance,
                                   int dim, float epsilon, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_batch_norm_compute_run<TYPE>(x, mean, variance, dim, epsilon, out); break; }
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
TS_REGISTER_OPERATOR(BatchNorm, GPU, name::layer::batch_norm())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(BatchNorm, ts::GPU, name::layer::batch_norm())
#endif
