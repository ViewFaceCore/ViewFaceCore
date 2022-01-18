#include "kernels/gpu/topkv2.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"
#include "backend/name.h"

#include <numeric>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <kernels/gpu/gpu_kernel.h>


namespace ts {
    namespace gpu {
        template <typename T>
        static __device__ void adjust_node(T *arr, int n, int len, int *arr2) {
            int l, r, max, index;
            T tmp;
            l = 2 * n + 1; 
            r = 2 * n + 2;
            max = n;

            if (l<len&&arr[l]>arr[n])
                max = l;
            if (r<len&&arr[r]>arr[max])
                max = r;
    
            if (max != n) {
                tmp = arr[n];
                arr[n] = arr[max];
                arr[max] = tmp;

                index = arr2[n];
                arr2[n] = arr2[max];
                arr2[max] = index; 
                adjust_node(arr, max, len, arr2);
            }
        }

        template <typename T>
        static __device__ void sort_heap(T *arr, int len, int *arr2) {
            for (int i = len / 2; i >= 0; i--)
                adjust_node(arr, i, len, arr2);
            int index;
            T   tmp;
            for (int i = len - 1; i >= 0; i--) {
                tmp = arr[0];
                arr[0] = arr[i];
                arr[i] = tmp;

                index = arr2[0];
                arr2[0] = arr2[i];
                arr2[i] = index;
                adjust_node(arr, 0, i, arr2);
            }
        }

        template<typename T>
        __global__ static void gpu_topkv2_kernel(const T* x_data, int32_t* psort, T *out_data, int size, int x_stride, int out_stride) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size)
                return;

            int out_index = index * out_stride;
            int x_index = index * x_stride;

            for(int i=0; i<out_stride; i++) {
                psort[i + out_index] = i;
                out_data[i + out_index] = x_data[i + x_index];
            }
          
 
            sort_heap<T>(out_data + out_index, out_stride, psort + out_index);
                
            for(int i=out_stride; i<x_stride; i++) {
                if(x_data[i + x_index ] < out_data[ out_index]) {
                    continue;
                }

                out_data[out_index] = x_data[i + x_index];
                psort[out_index] = i;
                sort_heap<T>(out_data + out_index, out_stride, psort + out_index); 
            }

        }

        template <typename T>
        static void gpu_topkv2_compute_run(const Tensor &x, int K, bool sorted, Tensor &values, Tensor &indices) {
            auto &x_shape = x.sizes();
            auto out = values;

            T * p_outdata = out.data<T>();
            const T* p_xdata  = x.data<T>();

            Shape out_shape = out.sizes();
            
            Tensor sort_tensor(out.device(), INT32, out_shape);
         
            int * psort = sort_tensor.data<int>(); 
            int number = out.count();
            int steps = number / out_shape[out_shape.size() - 1];
            int out_stride = out_shape[out_shape.size() - 1];
            int x_stride = x_shape[x_shape.size() - 1];

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(steps, blockSize.x));

            RUN_KERNEL(gpu_topkv2_kernel<T>, gridSize, blockSize,
                       p_xdata, psort, p_outdata, steps, x_stride, out_stride);

            indices = sort_tensor;
        }


        void Topkv2::topkv2(const Tensor &x, int K, bool sorted, Tensor &values, Tensor &indices) {
            DTYPE dtype = x.dtype();
           
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_topkv2_compute_run<TYPE>(x, K, sorted, values, indices); break; }
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

//using namespace ts;
//using namespace gpu;
//TS_REGISTER_OPERATOR(Topkv2, GPU, name::layer::topkv2())
//#ifdef TS_USE_CUDA_FP16
//TS_REGISTER_FP16_OPERATOR(Topkv2, GPU, name::layer::topkv2())
//#endif
