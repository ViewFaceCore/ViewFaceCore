#include <backend/base/base_norm_image.h>
#include <algorithm>

#include "kernels/gpu/operator_on_gpu.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"
#include "backend/name.h"

#include "kernels/gpu/math_gpu.h"
#include "kernels/gpu/gpu_kernel.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

namespace ts {
    namespace gpu {
        class NormImage : public OperatorOnGPU<base::NormImage> {
        public:
            using self = NormImage;
            using supper = OperatorOnGPU<base::NormImage>;

            void norm_image(const Tensor &x, float epsilon, Tensor &out) override;
        };
    }
}


namespace ts {
    namespace gpu {

        template<typename T>
        __global__ static void mean_kernel(const int N, T *x) {

            int index = blockDim.x * blockIdx.x + threadIdx.x;

            for (; index < 1; index += blockDim.x * gridDim.x) {
                x[index] /= N;
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__ void mean_kernel<half>(const int N, half *x) {

            int index = blockDim.x * blockIdx.x + threadIdx.x;
            half half_N = half(float(N));

            for (; index < 1; index += blockDim.x * gridDim.x) {
                x[index] /= half_N;
            }
        }
#endif

        template<typename T>
        __global__ static void dev_kernel(const int N, const T *x,T* mean, T * z) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            __shared__ T cache[CUDA_THREAD_NUM];

            int cache_index = threadIdx.x;
            T temp = T(0.f);
            for (; index < N; index += blockDim.x * gridDim.x) {
                T sub_tmp = x[index] - *mean;
                temp += sub_tmp * sub_tmp;
            }
            cache[cache_index] = temp;

            __syncthreads();

            unsigned int floor_pow = blockDim.x;
            if (floor_pow & (floor_pow - 1))
            {
                while (floor_pow & (floor_pow - 1))
                {
                    floor_pow &= (floor_pow - 1);
                }
                if (cache_index >= floor_pow)
                {
                    cache[cache_index - floor_pow] += cache[cache_index];
                }
                __syncthreads();
            }

            for (int i = floor_pow / 2; i > 0; i /= 2)
            {
                if (cache_index < i)
                {
                    cache[cache_index] += cache[cache_index + i];
                }
                __syncthreads();
            }

            if (cache_index == 0) {
                z[blockIdx.x] = cache[0];
            }
        }

        template<typename T>
        __global__ static void std_dev_kernel(const int N, T *x, T epsilon) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            for (; index < 1; index += gridDim.x * blockDim.x) {
                x[index] = sqrt(x[index] / N);
                // x[index] = max(x[index], T(1 / sqrt(T(N))));
                x[index] = T(1) / (x[index] + epsilon);
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__  void std_dev_kernel<half>(const int N, half *x, half epsilon) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            half half_N = half(float(N));
            half one(1.f);

            for (; index < 1; index += gridDim.x * blockDim.x) {
                x[index] = hsqrt(x[index] / half_N);
                half temp = one / hsqrt(half_N);
                // x[index] = x[index] > temp ? x[index] : temp;
                x[index] = one / (x[index] + epsilon);
            }
        }
#endif

        template<typename T>
        __global__ static void prewhiten_kernel(const int N, T *x, T* mean,T * dev_rec) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            for (; index < N; index += gridDim.x * blockDim.x) {
                x[index] -= *mean;
                x[index] *= *dev_rec;
            }

        }

        template<typename T>
        void cpu_norm_image_compute_run(const Tensor &x, float epsilon, Tensor &out, MemoryDevice& mem_device) {
            auto output_shape = out.sizes();
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();
            memcpy(output_data, out.device(), count * sizeof(T), input_data, x.device(), count * sizeof(T));
            //memcpy(output_data, input_data, count * sizeof(T));

            // fot batch
            int batch = x.size(0);
            count /= batch;
            auto batch_outout_data = output_data;

            int grid_size = CUDA_BLOCK(count, CUDA_THREAD_NUM);
            int block_size = CUDA_THREAD_NUM;

            Tensor buffer_tensor = Tensor(Tensor::InFlow::DEVICE, out.dtype(), {1 + 1 + block_size});
            T *mean = buffer_tensor.data<T>();
            T *std_dev = mean + 1;
            T *dev_buffer = std_dev + 1;

            T *at = nullptr;

            for (int n = 0; n < batch; ++n) {
                at = batch_outout_data;
                math<T>::sum(count, at,mean);
                RUN_KERNEL(mean_kernel<T>, 1, 1, count,mean);

                at = batch_outout_data;
                RUN_KERNEL(dev_kernel<T>, grid_size, block_size, count, at, mean, dev_buffer);
                math<T>::sum(grid_size, dev_buffer, std_dev);
                RUN_KERNEL(std_dev_kernel<T>, 1, 1, count, std_dev, T(epsilon));

                at = batch_outout_data;
                RUN_KERNEL(prewhiten_kernel<T>, grid_size, block_size, count,at,mean, std_dev);

                batch_outout_data += count;
            }
        }

        void NormImage::norm_image(const Tensor &x, float epsilon, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            auto running_mem_device = running_memory_device();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_norm_image_compute_run<TYPE>(x, epsilon, out, running_mem_device); break; }
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
TS_REGISTER_OPERATOR(NormImage, ts::GPU, name::layer::norm_image())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(NormImage, ts::GPU, name::layer::norm_image())
#endif
