#include <kernels/cpu/quantized/conv2d_quantized_core.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include "kernels/common/simd.h"
#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif

namespace ts {
    namespace cpu {

        template<typename T>
        static void cpu_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
                                           const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                           std::vector<float>dequantize_scales, Tensor &out, Stack &stack) {
            auto weight_shape = w.sizes();
            auto output_shape = out.sizes();
            auto x_shape = x.sizes();
            int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
            int conv_out_spatial_dim = output_shape[2] * output_shape[3];
            int output_number_offset = output_shape[1] * conv_out_spatial_dim;
            int input_number_offset = x_shape[1] * x_shape[2] * x_shape[3];
            int col_buffer_size = x_shape[1] * weight_shape[2] * weight_shape[3] * output_shape[2] * output_shape[3];

            auto number = x_shape[0];
            auto input_channels = x_shape[1];
            Size2D ksize(weight_shape[2], weight_shape[3]);
            Size2D input(x_shape[2], x_shape[3]);

            const T *pinput = x.data<T>();
            const int8_t *pweight = w.data<T>();
            float *poutput = out.data<float>(); 

            Tensor output_int32 = stack.make(INT32, out.sizes(), MemoryDevice(CPU));
            int32_t* poutput_int32 = output_int32.data<int32_t>();

            Tensor col_tensor;
            T *col_buffer = nullptr;

            bool is_1x1_conv = stride.height == 1 && stride.width == 1 &&
                               ksize.height == 1 && ksize.width == 1 &&
                               padding.top == 0 && padding.bottom == 0 &&
                               padding.left == 0 && padding.right == 0;

            // 1x1 conv do not need im2col
            if (!is_1x1_conv) {
                Shape col_shape;
                col_shape.resize(1);
                col_shape[0] = col_buffer_size;
                col_tensor = stack.make(x.dtype(), col_shape, MemoryDevice(CPU));
                col_buffer = col_tensor.data<T>();
            }

            for (int i = 0; i < number; i++) {
                if (is_1x1_conv) {
                    //std::memcpy(col_buffer,pinput,sizeof(T)*col_buffer_size);
                    col_buffer = const_cast<T *>(pinput);
                } else {
                    ::memset(col_buffer, 0, col_buffer_size * sizeof(T));
                    im2col_cpu(pinput, input_channels, input.height, input.width,
                               ksize.height, ksize.width,
                               padding.top, padding.bottom,
                               padding.left, padding.right,
                               stride.height, stride.width,
                               dilation.height, dilation.width,
                               col_buffer, T(padding_value));
                }
                cpu::math<T, int32_t>::gemm(ts::blas::NoTrans,ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
                               kernel_dims, 1, pweight, col_buffer, 0, poutput_int32);
                pinput += input_number_offset;
                poutput_int32 += output_number_offset;
            }

            //NOTE:fuse Dequantize(int32 to fp32) in conv2d_quantize now.
            auto input_data = output_int32.data<int32_t>();
            auto out_shape = out.sizes();
            int channal_offset = out_shape[2] * out_shape[3];
            int num_offset = out_shape[1] * channal_offset;
            for (int n = 0; n < out_shape[0]; n++){
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < out_shape[1]; c++){
                    auto input_cur = input_data + n * num_offset + c * channal_offset;
                    float dequantize_scale = dequantize_scales[c];
                    float32x4x2 dequantize_scale_x4x2(dequantize_scale);
                    int count = channal_offset;
                    int count_x8 = count >> 3;
                    int remain = count_x8 << 3;
                    for (int i = 0; i < count_x8; i++){
                        int ii = i * 8;
                        float32x4x2 input_x4x2 = intx4x2_to_float32x4x2(int32x4x2(&input_cur[ii]));
                        float32x4x2 out_x4x2 = input_x4x2 * dequantize_scale_x4x2;
                        out_x4x2.store(poutput);
                        poutput += 8;
                    }
                    for (int i = remain; i < count; i++){
                        *poutput++ = input_cur[i] * dequantize_scale;
                    }
                }
            }

//            auto input_data = output_int32.data<int32_t>();
//            int dequantize_group = dequantize_scales.size();
//            int output_count = out.count();
//            int loop_count = std::ceil(static_cast<float>(output_count) / dequantize_group);
//            int index = 0;
//#ifdef TS_USE_OPENMP
//            #pragma omp parallel for num_threads(openmp_threads())
//#endif
//            for (int n = 0; n < dequantize_group; n++) {
//                float dequantize_scale = dequantize_scales[n];
//                int loop_count_temp = loop_count;
//                while (index < output_count && loop_count_temp) {
//                    poutput[index] = input_data[index] * dequantize_scale;
//                    index++;
//                    loop_count_temp--;
//                }
//            }
        }

        void Conv2DQuantizedCore::conv2d(const Tensor &x, const Padding2D &padding, float padding_value, const Tensor &w,
                            const Stride2D &stride, const Dilation2D &dilation, Conv2DFormat format, 
                            std::vector<float>dequantize_scales,Tensor &out,Stack &stack) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "Conv2D_quantized only support NCHW" << eject;
            }
            DTYPE dtype = x.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, dequantize_scales, out, stack);; break; }
                //DECLARE_COMPUTE_RUN(FLOAT32, float);
                //DECLARE_COMPUTE_RUN(FLOAT64, double);
                DECLARE_COMPUTE_RUN(INT8, int8_t);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "Conv2D_quantized not support this data type: " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}
