#include "backend/base/base_reduce_mean.h"
#include "kernels/gpu/operator_on_gpu.h"

#include <core/tensor_builder.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>
#include <math.h>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/cuda_context.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"
#include "kernels/gpu/cudax_fp16_math.h"

#include "global/fp16_operator_factory.h"
#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {
        class ReduceMean : public OperatorOnGPU<base::ReduceMean> {
        public:
            using self = ReduceMean;
            using supper = OperatorOnGPU<base::ReduceMean>;

            void reduce(const Tensor &x, std::vector<int> dims, Tensor &out) override;
        };
    }
}

namespace ts {
    namespace gpu {
        template <typename T>
        static __global__ void reduce_sum_kernel(const T*input_data, T*output_data,
                                                 int input_count, int output_count,
                                                 int number, int channels, int width, int number_step) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < input_count) {
                auto i = index / number_step;
                auto w = index % width;
                auto local_input_data = input_data + index;
                auto local_output_data = output_data + i * width + w;
                atomicAdd(local_output_data, *local_input_data);
            }
        }
        template <typename T>
        static __global__ void reduce_sum_kernel_no_atomic(const T* input_data, T* output_data,
                                                           int channels, int number, int width) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int size = number * width;
            for (; index < size; index += blockDim.x * gridDim.x)
            {
                int n = index / width;
                int s = index % width;
                T sum = T(0.f);
                for (int k = 0; k < channels; k++)
                {
                    sum += input_data[(n * channels + k) * width + s];
                }
                output_data[index] = sum;
            }
        }

        template<typename T>
        __global__ static void mean_kernel(T* data, int size, T mean) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size)
            {
                data[index] /= mean;
            }
        }

        template <typename T>
        static void local_run_atomic_kernel(const T*input_data, T*output_data,
                                            int input_count, int output_count,
                                            int number, int channels, int width, int number_step, cudaStream_t stream) {
            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(input_count, blockSize.x));

            cudaMemsetAsync(output_data, 0, output_count * sizeof(T), stream);

            RUN_KERNEL_STREAM(reduce_sum_kernel<T>, gridSize, blockSize, 0, stream,
                              input_data, output_data,
                              input_count, output_count, number, channels, width,
                              channels * width);
            // mean
            RUN_KERNEL_STREAM(mean_kernel<T>, gridSize, blockSize, 0, stream,
                              output_data, output_count, T(channels));
        }

        template <typename T>
        static void local_run_kernel(const T*input_data, T*output_data,
                                     int input_count, int output_count,
                                     int number, int channels, int width, int number_step, cudaStream_t stream) {
            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(output_count, blockSize.x));
            RUN_KERNEL_STREAM(reduce_sum_kernel_no_atomic<T>, gridSize, blockSize, 0, stream,
                              input_data, output_data,
                              channels, number, width);
            // mean
            RUN_KERNEL_STREAM(mean_kernel<T>, gridSize, blockSize, 0, stream,
                              output_data, output_count, T(channels));
        }

#ifdef TS_USE_CUDA_FP16
        template <>
        __global__ void reduce_sum_kernel_no_atomic<half>(const half* input_data, half* output_data,
                                                          int channels, int number, int width) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int size = number * width;
            for (; index < size; index += blockDim.x * gridDim.x)
            {
                int n = index / width;
                int s = index % width;
                half sum = __float2half(0.f);
                for (int k = 0; k < channels; k++)
                {
                    sum = sum + input_data[(n * channels + k) * width + s];
                }
                output_data[index] = sum;
            }
        }

        template <>
        void local_run_kernel<half>(const half*input_data, half*output_data,
                                     int input_count, int output_count,
                                     int number, int channels, int width, int number_step, cudaStream_t stream) {
            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(output_count, blockSize.x));
            RUN_KERNEL_STREAM(reduce_sum_kernel_no_atomic<half>, gridSize, blockSize, 0, stream,
                              input_data, output_data,
                              channels, number, width);
            // mean
            RUN_KERNEL_STREAM(mean_kernel<half>, gridSize, blockSize, 0, stream,
                              output_data, output_count, __float2half(float(channels)));
        }
#endif

        template<typename T>
        void gpu_reduce_mean_compute_run(const Tensor &x, std::vector<int> dims, Tensor &out) {
            int dims_size = int(dims.size());
            auto &size = x.sizes();
            auto number = std::accumulate(size.begin(), size.begin() + dims[0], 1, std::multiplies<int32_t>());
            int channels = 1;
            for (int i = 0; i < dims_size; i++){
                channels *= size[dims[i]];
            }
            auto width = std::accumulate(size.begin() + dims[dims_size - 1] + 1, size.end(), 1, std::multiplies<int32_t>());

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();
            auto input_count = x.count();
            auto output_count = out.count();

            auto &context = ctx::ref<DeviceContext>();
            CUDAContextHandle *handle = reinterpret_cast<CUDAContextHandle *>(context.handle);
            auto cuda_stream = handle->stream();

            local_run_kernel<T>(input_data, output_data,
                                input_count, output_count, number, channels, width,
                                channels * width, cuda_stream);
        }

        void ReduceMean::reduce(const Tensor &x, std::vector<int> dims, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_reduce_mean_compute_run<TYPE>(x, dims, out); break; }
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
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(ReduceMean, ts::GPU, name::layer::reduce_mean())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(ReduceMean, ts::GPU, name::layer::reduce_mean())
#endif

