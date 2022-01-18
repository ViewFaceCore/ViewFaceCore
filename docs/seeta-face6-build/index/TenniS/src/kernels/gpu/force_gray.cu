//
// Created by kier on 19-6-27.
//

#include "backend/base/base_force_gray.h"
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
        namespace {
            template<typename T, int W>
            class __gpu_array {
            public:
                using Type = T;
                Type data[W];

                __gpu_array() = default;

                template<typename Beg, typename End>
                __host__ __gpu_array(Beg beg, End end) {
                    size_t i = 0;
                    while (beg != end) {
                        data[i] = *beg;
                        ++i;
                        ++beg;
                    }
                }

                __host__ __device__ Type &operator[](int i) { return data[i]; }
            };

            template <typename T>
            class __sum_type {
            public:
                using Type = float;
            };

            template <>
            class __sum_type<double> {
            public:
                using Type = double;
            };
        }

        template<typename T, typename S, int W>
        __global__ static void force_gray_kernel(const T* input_data, T* output_data, int size, int input_channels, int output_channels, __gpu_array<S, W> scale) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < size)
            {
                auto input_pixel = &input_data[i * input_channels];
                auto output_pixel = &output_data[i * output_channels];

                typename __sum_type<T>::Type pixel = 0;

                for (int j = 0; j < input_channels; ++j) {
                    pixel += scale[j] * input_pixel[j];
                }

                output_pixel[0] = T(pixel);
            }
        }

        template<typename T, typename S>
        __global__ static void force_gray_kernel(const T* input_data, T* output_data, int size, int input_channels, int output_channels, const S* scale) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < size)
            {
                auto input_pixel = &input_data[i * input_channels];
                auto output_pixel = &output_data[i * output_channels];

                typename __sum_type<T>::Type pixel = 0;

                for (int j = 0; j < input_channels; ++j) {
                    pixel += scale[j] * input_pixel[j];
                }

                output_pixel[0] = T(pixel);
            }
        }

        template<typename T>
        void gpu_force_gray_compute_run(const Tensor &x, const std::vector<float> &scale, Tensor &out) {
            auto &size = x.sizes();
            auto dims = x.dims();
            auto number = std::accumulate(size.begin(), size.end() - 1, 1, std::multiplies<int32_t>());
            auto input_channels = x.size(dims - 1);
            auto output_channels = out.size(dims - 1);

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(number, blockSize.x));

            switch (scale.size()) {
                default: {
                    auto gpu_scale_data_size = scale.size() * sizeof(float);
                    Tensor gpu_scale(Tensor::InFlow::DEVICE, FLOAT32, {int32_t(scale.size())});
                    memcpy(gpu_scale.data(), gpu_scale.device(), gpu_scale_data_size,
                            scale.data(), Device(CPU), gpu_scale_data_size);

                    RUN_KERNEL(force_gray_kernel<T>, gridSize, blockSize,
                               input_data, output_data, number, input_channels,
                               output_channels, gpu_scale.data<float>());
                    break;
                }
#define CASE_N_FORCE_KERNEL(N) \
                case N: { \
                    __gpu_array<float, N> gpu_scale(scale.begin(), scale.end()); \
                    gpu_scale[0]; \
                    RUN_KERNEL(force_gray_kernel<T>, gridSize, blockSize, \
                               input_data, output_data, number, input_channels, output_channels, gpu_scale); \
                    break; \
                }
                CASE_N_FORCE_KERNEL(1)
                CASE_N_FORCE_KERNEL(2)
                CASE_N_FORCE_KERNEL(3)
                CASE_N_FORCE_KERNEL(4)
                CASE_N_FORCE_KERNEL(5)
                CASE_N_FORCE_KERNEL(6)
                CASE_N_FORCE_KERNEL(7)
                CASE_N_FORCE_KERNEL(8)
#undef CASE_N_FORCE_KERNEL
            }
        }

        class ForceGray : public OperatorOnGPU<base::ForceGray> {
        public:
            void force_gray(const Tensor &x, const std::vector<float> &scale, Tensor &out) final {
                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_force_gray_compute_run<TYPE>(x, scale, out); break; }
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
TS_REGISTER_OPERATOR(ForceGray, GPU, name::layer::force_gray())