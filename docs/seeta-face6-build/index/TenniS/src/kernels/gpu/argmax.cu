#include "kernels/gpu/argmax.h"
#include "global/operator_factory.h"
//#include "global/fp16_operator_factory.h"
#include "backend/name.h"

#include <numeric>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <kernels/gpu/gpu_kernel.h>


namespace ts {
    namespace gpu {

        template<typename T>
        __global__ static void gpu_argmax_kernel(const T* x_data, int32_t* out_data, int size, int width, int input_width, int axis) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size)
                return; 
           
            int32_t *ptr = out_data + index * width;
            int cur_input_width = index * input_width;
            T tmp;
            int32_t cur_index = 0;
            for(int k=0; k<width; k++) {
                for(int m=0; m<axis; m++) {
                    int cur_step = m * width;
                    if(m == 0) {
                        cur_index = 0;
                        tmp = *(x_data + k + cur_step + cur_input_width);
                    }else if(*(x_data + k + cur_step + cur_input_width) > tmp) {
                        tmp = *(x_data + k + cur_step + cur_input_width);
                        cur_index = m;
                    }
                }

                *(ptr + k) = cur_index;//tmp;
            }
             
        }


        template <typename T>
        static void gpu_argmax_compute_run(const Tensor &x, int axis,  Tensor &out) {
            auto &x_shape = x.sizes();
            if(axis < 0) {
                axis += int(x_shape.size());
            }

            auto number = std::accumulate(x_shape.begin(), x_shape.begin() + axis, 1, std::multiplies<int>());
            auto width = std::accumulate(x_shape.begin() + axis + 1, x_shape.end(), 1, std::multiplies<int>());
            int input_width = width * x_shape[axis];


            const T* x_data = x.data<T>();
            int32_t * out_data = out.data<int32_t>();

            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(number, blockSize.x));

            RUN_KERNEL(gpu_argmax_kernel<T>, gridSize, blockSize,
                       x_data, out_data, number, width, input_width, x_shape[axis]);
        }


        void ArgMax::argmax(const Tensor &x, int dim, Tensor &out) {
            DTYPE dtype = out.dtype();
           
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_argmax_compute_run<TYPE>(x, dim, out); break; }
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
TS_REGISTER_OPERATOR(ArgMax, GPU, name::layer::argmax())
#ifdef TS_USE_CUDA_FP16
//TS_REGISTER_FP16_OPERATOR(ArgMax, ts::GPU, name::layer::argmax())
#endif
