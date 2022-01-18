#include <kernels/gpu/spacetobatch4d.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>

#include "kernels/gpu/gpu_kernel.h"

#include <cuda_fp16.h>

//#include "kernels/common/simd.h"

namespace ts {
    namespace gpu {

        template<typename T>
        __global__ static void gSpaceToBatchND_kernel(const T *input_data, T *output_data,
                        const int input_number, const int input_channels, const int input_height, const int input_width,
                        const int output_number, const int output_channels, const int output_height, const int output_width,
                        const int input_size, const int input_number_step, const int input_channels_step, const int input_height_step,
                        const int output_size, const int output_number_step, const int output_channels_step, const int output_height_step,
                        const int block_height, const int block_width, const int padding_top, const int padding_bottom, const int padding_left, const int padding_right) {

            int index = threadIdx.x + blockIdx.x * blockDim.x;

            if (index < input_size)
            {
                int at_input_i = index;

                int n = index / input_number_step;
                index %= input_number_step;
                int c = index / input_channels_step;
                index %= input_channels_step;
                int h = index / input_height_step;
                index %= input_height_step;
                int w = index;

                int on = ((h + padding_top) % block_height * block_width + (w + padding_top) % block_width) * input_number + n;
                int oc = c;
                int oh = (h + padding_top) / block_height;
                int ow = (w + padding_left) / block_width;

                int at_output_i = on * output_number_step
                                                  + oc * output_channels_step
                                                  + oh * output_height_step
                                                  + ow;

                output_data[at_output_i] = input_data[at_input_i];
            }

        }


        template<typename T>
        static void gpu_spacetobatch4d_compute_run(const Tensor &x, const int padding_top, const int padding_bottom,
                    const int padding_left,const int padding_right, const int block_height, const int block_width, Tensor &out) { 

            Shape x_shape = x.sizes();
            Shape out_shape = out.sizes();

            int input_number = x_shape[0];
            int input_channels = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];

            int output_number = out_shape[0];
            int output_channels = out_shape[1];
            int output_height = out_shape[2];
            int output_width = out_shape[3];

            int input_size = input_number * input_channels * input_height * input_width;
            int input_number_step = input_channels * input_height * input_width;
            int input_channels_step = input_height * input_width;
            int input_height_step = input_width;
            // int input_width_step = 1;

            int output_size = output_number * output_channels * output_height * output_width;
            int output_number_step = output_channels * output_height * output_width;
            int output_channels_step = output_height * output_width;
            int output_height_step = output_width;
            // int output_width_step = 1;

            const T * pinput = x.data<T>();
            T * poutput = out.data<T>();
            T n = T(0.f);
            memset(poutput, out.device(), out.count() * sizeof(T), &n, MemoryDevice(CPU), sizeof(T));

            RUN_KERNEL(gSpaceToBatchND_kernel<T>, CUDA_BLOCK(input_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pinput, poutput,
                       input_number, input_channels, input_height, input_width,
                       output_number, output_channels, output_height, output_width,
                       input_size, input_number_step, input_channels_step, input_height_step,
                       output_size, output_number_step, output_channels_step, output_height_step,
                       block_height, block_width, padding_top, padding_bottom, padding_left, padding_right);

        }


        void SpaceToBatch4D::spacetobatch4d_run(const Tensor &x,const int padding_top, const int padding_bottom,
                    const int padding_left,const int padding_right, const int block_height, const int block_width, Tensor &out){
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_spacetobatch4d_compute_run<TYPE>(x, padding_top,padding_bottom,padding_left,padding_right,block_height,block_width, out); break; }
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
TS_REGISTER_OPERATOR(SpaceToBatch4D, GPU, name::layer::spacetobatch4d())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(SpaceToBatch4D, GPU, name::layer::spacetobatch4d())
#endif
