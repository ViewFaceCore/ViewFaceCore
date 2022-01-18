#include <backend/base/base_l2_norm.h>

#include <kernels/gpu/operator_on_gpu.h>
#include <core/tensor_builder.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"
//#include <algorithm>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/gpu_kernel.h"

#include "kernels/gpu/cudax_fp16_math.h"


namespace ts {
    namespace gpu {
        class L2Norm : public OperatorOnGPU<base::L2Norm> {
        public:
            using self = L2Norm;
            using supper = OperatorOnGPU<base::L2Norm>;

            void normalize(const Tensor &x, int dim, float epsilon, Tensor &out) override;
        };
    }
}


namespace ts {
    namespace gpu {
        template<typename T>
        __global__ static void square_kernel(const T *input_data, T *output_data, int count)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                output_data[index] = input_data[index] * input_data[index];
            }
        }

        template<typename T>
        __global__ static void sum_kernel(const T* input_data, T* output_data, int dim_num, int outer_num, int inner_num)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int size = outer_num * inner_num;
            for (; index < size; index += blockDim.x * gridDim.x)
            {
                int n = index / inner_num;
                int s = index % inner_num;
                T sum = T(0.f);
                for (int k = 0; k < dim_num; k++)
                {
                    sum += input_data[(n * dim_num + k) * inner_num + s];
                }
                output_data[index] = sum;
            }
        }

        template<typename T>
        __global__ static void div_kernel(const T *input_data, const T* scale_data, T* output_data, int count, int dim_num, int outer_num, int inner_num,
                T epsilon)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                int n = index / dim_num / inner_num;
                int s = index % inner_num;
                output_data[index] = input_data[index] / sqrt(scale_data[n * inner_num + s] + epsilon);
            }
        }

        template <typename T>
        class float_to {
        public:
            static T F(float a) { return T(a); }
        };


#ifdef TS_USE_CUDA_FP16
        template <>
        class float_to<half> {
        public:
            static half F(float a) { return __float2half(a); }
        };
#endif

        template<typename T>
        void cpu_l2_norm_compute_run(const Tensor &x, int m_dim, float epsilon, Tensor &out, MemoryDevice& mem_device) {
            auto output_shape = out.sizes();

            int pre_num = 1;
            for (int i = 0; i < m_dim; i++) {
                pre_num *= output_shape[i];
            }
            int inner_num = 1;
            for (int i = m_dim + 1; i < output_shape.size(); i++) {
                inner_num *= output_shape[i];
            }

            int axis = output_shape[m_dim];

            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();

            int count = out.count();
            // memcpy(output_data, out.device(), count * sizeof(T), input_data, x.device(), count * sizeof(T));

            int scale_data_size = out.count() / axis;

            Shape scale_shape;
            scale_shape.resize(1);
            scale_shape[0] = scale_data_size;
            Tensor scale_tensor(Tensor::InFlow::DEVICE, out.dtype(), scale_shape);
            T *scale_data = scale_tensor.data<T>();

            dim3 block_size(CUDA_THREAD_NUM);

            dim3 square_kernel_grid_size((count + block_size.x - 1) / block_size.x);
            RUN_KERNEL(square_kernel<T>, square_kernel_grid_size, block_size, input_data, output_data, count);

            dim3 sum_kernel_grid_size((pre_num * inner_num + block_size.x - 1) / block_size.x);
            RUN_KERNEL(sum_kernel<T>, sum_kernel_grid_size, block_size, output_data, scale_data, axis, pre_num, inner_num);

            dim3 div_kernel_grid_size((count + block_size.x - 1) / block_size.x);
            RUN_KERNEL(div_kernel<T>, div_kernel_grid_size,block_size, input_data, scale_data, output_data, count,axis,pre_num,inner_num, float_to<T>::F(epsilon));
        }

        void L2Norm::normalize(const Tensor &x, int dim, float epsilon, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            auto running_mem_device = running_memory_device();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_l2_norm_compute_run<TYPE>(x, dim, epsilon, out, running_mem_device); break; }
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
TS_REGISTER_OPERATOR(L2Norm, ts::GPU, name::layer::l2_norm())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(L2Norm, ts::GPU, name::layer::l2_norm())
#endif
