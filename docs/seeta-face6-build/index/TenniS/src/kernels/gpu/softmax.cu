#include <kernels/gpu/softmax.h>
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
        template<typename T>
        __global__ static void max_kernel(const T* input_data, T* scale_data, int dim_num, int outer_num, int inner_num)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int size = outer_num * inner_num;
            for (; index < size; index += blockDim.x * gridDim.x)
            {
                int n = index / inner_num;
                int s = index % inner_num;
                T max_val = input_data[n * dim_num * inner_num + s];
                for (int k = 1; k < dim_num; k++)
                {
                    max_val = max(input_data[(n * dim_num + k) * inner_num + s], max_val);
                }   
                scale_data[index] = max_val;
            }
        }

#ifdef TS_USE_CUDA_FP16
        template<>
        __global__ void max_kernel<half>(const half* input_data, half* scale_data, int dim_num, int outer_num, int inner_num)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int size = outer_num * inner_num;
            for (; index < size; index += blockDim.x * gridDim.x)
            {
                int n = index / inner_num;
                int s = index % inner_num;
                half max_val = input_data[n * dim_num * inner_num + s];
                for (int k = 1; k < dim_num; k++)
                {
                    half input_cur = input_data[(n * dim_num + k) * inner_num + s];
                    max_val = input_cur > max_val ? input_cur : max_val;
                }
                scale_data[index] = max_val;
            }
        }
#endif

        template<typename T>
        __global__ static void substract_kernel(const T* scale_data,T* output_data,int count,int dim_num, int outer_num, int inner_num)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                int n = index / dim_num / inner_num;
                int s = index % inner_num;
                output_data[index] -= scale_data[n * inner_num + s];
            }
        }

        template<typename T>
        __global__ static void exp_kernel(const T* input_data, T* output_data, int count)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                output_data[index] = exp(input_data[index]);
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
        __global__ static void div_kernel(const T* input_data, T* output_data, int count, int dim_num, int outer_num, int inner_num)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                int n = index / dim_num / inner_num;
                int s = index % inner_num;
                output_data[index] /= input_data[n * inner_num + s];
            }
        }

        template<typename T>
        void cpu_softmax_compute_run(const Tensor &x, int m_dim, bool m_smooth, Tensor &out, MemoryDevice& mem_device) {
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
            memcpy(output_data, out.device(), count * sizeof(T), input_data, x.device(), count * sizeof(T));

            int scale_data_size = out.count() / axis;

            

            Shape scale_shape;
            scale_shape.resize(1);
            scale_shape[0] = scale_data_size;
            Tensor scale_tensor(Tensor::InFlow::DEVICE, out.dtype(), scale_shape);
            T *scale_data = scale_tensor.data<T>();

            dim3 block_size(CUDA_THREAD_NUM);

            if (m_smooth)
            {
                dim3 max_kernel_grid_size((pre_num * inner_num + block_size.x - 1) / block_size.x);
                RUN_KERNEL(max_kernel<T>, max_kernel_grid_size, block_size, input_data, scale_data, axis, pre_num, inner_num);

                dim3 substract_kernel_grid_size((count + block_size.x - 1) / block_size.x);
                RUN_KERNEL(substract_kernel<T>, substract_kernel_grid_size, block_size, scale_data, output_data, count, axis, pre_num, inner_num);

            }

            dim3 exp_kernel_grid_size((count + block_size.x - 1) / block_size.x);
            RUN_KERNEL(exp_kernel<T>, exp_kernel_grid_size, block_size, output_data, output_data, count);

            dim3 sum_kernel_grid_size((pre_num * inner_num + block_size.x - 1) / block_size.x);
            RUN_KERNEL(sum_kernel<T>, sum_kernel_grid_size, block_size, output_data, scale_data, axis, pre_num, inner_num);

            dim3 div_kernel_grid_size((count + block_size.x - 1) / block_size.x);
            RUN_KERNEL(div_kernel<T>, div_kernel_grid_size,block_size, scale_data,output_data,count,axis,pre_num,inner_num);
        }

        void Softmax::softmax(const Tensor &x, int dim, bool smooth, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            auto running_mem_device = running_memory_device();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_softmax_compute_run<TYPE>(x, dim, smooth, out, running_mem_device); break; }
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
TS_REGISTER_OPERATOR(Softmax, ts::GPU, name::layer::softmax())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Softmax, ts::GPU, name::layer::softmax())
#endif
