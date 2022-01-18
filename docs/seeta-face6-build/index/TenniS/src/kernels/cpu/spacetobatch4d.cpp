#include <kernels/cpu/spacetobatch4d.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>

#include "kernels/common/simd.h"

namespace ts {
    namespace cpu {

        template<typename T>
        static void cpu_spacetobatch4d_compute_run(const Tensor &x, const int padding_top, const int padding_bottom,
                    const int padding_left,const int padding_right, const int block_height, const int block_width, Tensor &out) {

            Shape x_shape = x.sizes();
            Shape out_shape = out.sizes();

            int input_number = x_shape[0];
            int input_channels = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];

            // int output_number = out_shape[0];
            int output_channels = out_shape[1];
            int output_height = out_shape[2];
            int output_width = out_shape[3];

            // int input_number_step = input_channels * input_height * input_width;
            // int input_channels_step = input_height * input_width;
            // int input_height_step = input_width;
            // int input_width_step = 1;

            // int output_size = output_number * output_channels * output_height * output_width;
            int output_number_step = output_channels * output_height * output_width;
            int output_channels_step = output_height * output_width;
            int output_height_step = output_width;
            // int output_width_step = 1;

            const T *pinput = x.data<T>();
            T *poutput = out.data<T>();

            std::memset(poutput, 0, out.count() * sizeof(T));

            int at_input_i = 0; // n * input_number_step + c * input_channels_step;
            for (int n = 0; n < input_number; ++n) {
                for (int c = 0; c < input_channels; ++c) {
                    const int oc = c;
                    for (int h = 0; h < input_height; ++h) {
                        const int oh = (h + padding_top) / block_height;
                        const int pre_compute = (h + padding_top) % block_height * block_width;
                        for (int w = 0; w < input_width; ++w) {
                            const int on = (pre_compute + (w + padding_left) % block_width) * input_number + n;
                            const int ow = (w + padding_left) / block_width;

                            const int at_output_i = on * output_number_step
                                                    + oc * output_channels_step
                                                    + oh * output_height_step
                                                    + ow;
                            poutput[at_output_i] = pinput[at_input_i];

                            ++at_input_i;
                        }
                    }
                }
            }
        }


        void SpaceToBatch4D::spacetobatch4d_run(const Tensor &x,const int padding_top, const int padding_bottom,
                    const int padding_left,const int padding_right, const int block_height, const int block_width, Tensor &out){
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_spacetobatch4d_compute_run<TYPE>(x, padding_top,padding_bottom,padding_left,padding_right,block_height,block_width, out); break; }
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
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(SpaceToBatch4D, CPU, name::layer::spacetobatch4d())
