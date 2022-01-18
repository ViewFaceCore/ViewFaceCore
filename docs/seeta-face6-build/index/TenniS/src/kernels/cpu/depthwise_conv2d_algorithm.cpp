#include "kernels/cpu/depthwise_conv2d_algorithm.h"
#include "kernels/common/simd.h"

#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif

namespace ts {
    namespace cpu{

        template<typename T>
        inline static void depthwise_inner(
            const T* input_at,
            const T* kernel_at,
            const int output_h_index,
            const int output_w_index,
            const Padding2D & padding,
            float padding_value,
            const Stride2D &stride,
            const int input_h,
            const int input_w,
            const int output_width,
            T* output_at) {
            T sum = 0;
            int h_in = output_h_index * stride.height - padding.top;
            int w_in = output_w_index * stride.width - padding.left;
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j++){
                    int h_index = i + h_in;
                    int w_index = j + w_in;
                    if (h_index >= 0 && h_index < input_h && w_index >= 0 && w_index < input_w) {
                        sum += input_at[h_index * input_w + w_index] * (*kernel_at);
                    }
                    else {
                        sum += padding_value * (*kernel_at);
                    }
                    kernel_at++;
                }
            }
            output_at[output_h_index * output_width + output_w_index] = sum;
        }

        template<typename T>
        void DepthwiseConv2dAlgorithm<T>::depthwise_general(
            const Tensor & x,
            const Padding2D & padding,
            float padding_value,
            const Tensor & weight,
            const Stride2D &stride,
            const Dilation2D & dilation,
            Tensor & out){
            auto weight_shape = weight.sizes();
            auto output_shape = out.sizes();
            auto input_shape = x.sizes();

            const T *pinput = x.data<T>();
            const T *pweight_base = weight.data<T>();
            T *poutput = out.data<T>();
            for (int n = 0; n < output_shape[0]; n++) {
                for (int c = 0; c < output_shape[1]; c++) {
                    for (int h = 0; h < output_shape[2]; h++) {
                        for (int w = 0; w < output_shape[3]; w++) {
                            const T *pweight = pweight_base + c * weight_shape[2] * weight_shape[3];
                            T value = 0;
                            for (int kh = 0; kh < weight_shape[2]; kh++) {
                                for (int kw = 0; kw < weight_shape[3]; kw++) {
                                    int h_in = -padding.top + h * stride.height + kh * dilation.height;
                                    int w_in = -padding.left + w * stride.width + kw * dilation.width;
                                    if ((h_in >= 0) && (h_in < input_shape[2]) && (w_in >= 0) &&
                                        (w_in < input_shape[3])) {
                                        int offset = ((n * output_shape[1] + c) * input_shape[2] + h_in) * input_shape[3] + w_in;
                                        value += (*pweight) * pinput[offset];
                                    }
                                    else {
                                        value += (*pweight) * padding_value;
                                    }
                                    ++pweight;
                                }
                            }
                            *poutput++ = value;
                        }
                    }
                }
            }
        }

        template<typename T>
        void DepthwiseConv2dAlgorithm<T>::depthwise_3x3_s1(
            const Tensor &x,
            const Padding2D &padding,
            float padding_value,
            const Tensor &weight,
            const Stride2D &stride,
            const Dilation2D &dilation,
            Tensor &out) {
        }

        template<>
        void DepthwiseConv2dAlgorithm<float>::depthwise_3x3_s1(
            const Tensor &x,
            const Padding2D &padding,
            float padding_value,
            const Tensor &weight,
            const Stride2D &stride,
            const Dilation2D &dilation,
            Tensor &out) {

            // auto weight_shape = weight.sizes();
            auto output_shape = out.sizes();
            auto input_shape = x.sizes();
            int input_height = input_shape[2];
            int input_width = input_shape[3];
            int output_height = output_shape[2];
            int output_width = output_shape[3];

            int input_channel_offset = input_shape[2] * input_shape[3];
            int input_num_offset = input_shape[1] * input_channel_offset;
            int output_channel_offset = output_shape[2] * output_shape[3];
            int output_num_offset = output_shape[1] * output_channel_offset;
            
            //h_start = padding.top == 0 ? 0 : (padding.top - 1) / stride.height + 1;
            //h_end = padding.bottom == 0 ? output_height : output_height - ((padding.bottom - 1) / stride.height + 1)
            //w_strst = padding.left == 0 ? 0 : (padding.left - 1) / stride.width + 1
            //w_end = padding.right == 0 ? output_width : output_width - ((padding.right - 1) / stride.width + 1)
            int h_start = padding.top == 0 ? 0 : padding.top;
            int h_end = padding.bottom == 0 ? output_height : output_height - padding.bottom;
            int w_start = padding.left == 0 ? 0 : padding.left;
            int w_end = padding.right == 0 ? output_width : output_width - padding.right;
            
            const float* pinput = x.data<float>();
            const float* pkernel = weight.data<float>();
            float *poutput = out.data<float>();

            for (int n = 0; n < input_shape[0]; n++){
#ifdef TS_USE_OPENMP
//Note:Using both openmp and neon on armv7 could cause crashes.
#ifdef TS_ON_ARMV7
#else
#pragma omp parallel for num_threads(openmp_threads())
#endif
#endif
                for (int c = 0; c < output_shape[1]; c++){
                    const float* input_at = pinput + n * input_num_offset + c * input_channel_offset;
                    const float* kernel_at = pkernel + c * 9;

                    float* out_at = poutput + n * output_num_offset + c * output_channel_offset;
//#ifdef TS_USE_AVX
//                    float32x4x2 k0_x4(kernel_at);
//                    float32x4x2 k1_x4(kernel_at + 1);
//#else
                    float32x4 k0_x4(kernel_at);
                    float32x4 k1_x4(kernel_at + 3);
                    float32x4 k2_x4(kernel_at + 6);
//#endif

                    int h, w;

                    for (h = 0; h < h_start; h++){
                        for (w = 0; w < output_width; w++){
                            depthwise_inner(input_at, kernel_at,
                                            h, w, 
                                            padding, padding_value, stride,
                                            input_height, input_width, 
                                            output_width, out_at);
                        }
                    }

                    for (h = h_start; h + 1< h_end; h += 2){
                        for (w = 0; w < w_start; w++){
                            depthwise_inner(input_at, kernel_at,
                                            h, w,
                                            padding, padding_value, stride,
                                            input_height, input_width,
                                            output_width, out_at);

                            depthwise_inner(input_at, kernel_at,
                                            h + 1, w,
                                            padding, padding_value, stride,
                                            input_height, input_width,
                                            output_width, out_at);
                        }

                        for (w = w_start; w + 3 < w_end; w += 4){
                            float* out0 = out_at + h * output_width + w;
                            float* out1 = out0 + output_width;
                            float32x4 out0_x4(0.f), out1_x4(0.f);

                            const float* i0 = input_at + (h - padding.top) * input_width + (w - padding.left);
                            const float* i1 = i0 + input_width;
                            const float* i2 = i1 + input_width;
                            const float* i3 = i2 + input_width;
                            
                            //Note:i don't optimize concat by avx or sse now. 
#ifdef TS_USE_NEON
                            float32x4 i00_x4(i0), i00_tmp_x4(i0 + 4);
                            float32x4 i10_x4(i1), i10_tmp_x4(i1 + 4);
                            float32x4 i20_x4(i2), i20_tmp_x4(i2 + 4);
                            float32x4 i30_x4(i3), i30_tmp_x4(i3 + 4);

                            float32x4 i01_x4 = concat(i00_x4, i00_tmp_x4, 1);
                            float32x4 i02_x4 = concat(i00_x4, i00_tmp_x4, 2);
                            float32x4 i11_x4 = concat(i10_x4, i10_tmp_x4, 1);
                            float32x4 i12_x4 = concat(i10_x4, i10_tmp_x4, 2);
                            float32x4 i21_x4 = concat(i20_x4, i20_tmp_x4, 1);
                            float32x4 i22_x4 = concat(i20_x4, i20_tmp_x4, 2);
                            float32x4 i31_x4 = concat(i30_x4, i30_tmp_x4, 1);
                            float32x4 i32_x4 = concat(i30_x4, i30_tmp_x4, 2);
#else
                            float32x4 i00_x4(i0), i01_x4(i0 + 1), i02_x4(i0 + 2);
                            float32x4 i10_x4(i1), i11_x4(i1 + 1), i12_x4(i1 + 2);
                            float32x4 i20_x4(i2), i21_x4(i2 + 1), i22_x4(i2 + 2);
                            float32x4 i30_x4(i3), i31_x4(i3 + 1), i32_x4(i3 + 2);
#endif

                            out0_x4 = fmadd(i00_x4, k0_x4, out0_x4, 0);
                            out0_x4 = fmadd(i01_x4, k0_x4, out0_x4, 1);
                            out0_x4 = fmadd(i02_x4, k0_x4, out0_x4, 2);
                            out0_x4 = fmadd(i10_x4, k1_x4, out0_x4, 0);
                            out0_x4 = fmadd(i11_x4, k1_x4, out0_x4, 1);
                            out0_x4 = fmadd(i12_x4, k1_x4, out0_x4, 2);
                            out0_x4 = fmadd(i20_x4, k2_x4, out0_x4, 0);
                            out0_x4 = fmadd(i21_x4, k2_x4, out0_x4, 1);
                            out0_x4 = fmadd(i22_x4, k2_x4, out0_x4, 2);

                            out1_x4 = fmadd(i10_x4, k0_x4, out1_x4, 0);
                            out1_x4 = fmadd(i11_x4, k0_x4, out1_x4, 1);
                            out1_x4 = fmadd(i12_x4, k0_x4, out1_x4, 2);
                            out1_x4 = fmadd(i20_x4, k1_x4, out1_x4, 0);
                            out1_x4 = fmadd(i21_x4, k1_x4, out1_x4, 1);
                            out1_x4 = fmadd(i22_x4, k1_x4, out1_x4, 2);
                            out1_x4 = fmadd(i30_x4, k2_x4, out1_x4, 0);
                            out1_x4 = fmadd(i31_x4, k2_x4, out1_x4, 1);
                            out1_x4 = fmadd(i32_x4, k2_x4, out1_x4, 2);

                            out0_x4.store(out0); out1_x4.store(out1);
                        }

                        for (; w < output_width; w++) {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);

                            depthwise_inner(input_at, kernel_at,
                                h + 1, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }
                    for ( ; h < output_height; h++){
                        for ( w = 0; w < output_width; w++){
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }
                }
            }
        }

        template<typename T>
        void DepthwiseConv2dAlgorithm<T>::depthwise_3x3_s2(
            const Tensor &x,
            const Padding2D &padding,
            float padding_value,
            const Tensor &weight,
            const Stride2D &stride,
            const Dilation2D &dilation,
            Tensor &out) {
        }

        template<>
        void DepthwiseConv2dAlgorithm<float>::depthwise_3x3_s2(
            const Tensor &x,
            const Padding2D &padding,
            float padding_value,
            const Tensor &weight,
            const Stride2D &stride,
            const Dilation2D &dilation,
            Tensor &out) {

            auto weight_shape = weight.sizes();
            auto output_shape = out.sizes();
            auto input_shape = x.sizes();
            int input_height = input_shape[2];
            int input_width = input_shape[3];
            int output_height = output_shape[2];
            int output_width = output_shape[3];

            int input_channel_offset = input_shape[2] * input_shape[3];
            int input_num_offset = input_shape[1] * input_channel_offset;
            int output_channel_offset = output_shape[2] * output_shape[3];
            int output_num_offset = output_shape[1] * output_channel_offset;

            //h_start = padding.top == 0 ? 0 : (padding.top - 1) / stride.height + 1;
            //h_end = padding.bottom == 0 ? output_height : output_height - ((padding.bottom - 1) / stride.height + 1)
            //w_strst = padding.left == 0 ? 0 : (padding.left - 1) / stride.width + 1
            //w_end = padding.right == 0 ? output_width : output_width - ((padding.right - 1) / stride.width + 1)
            int h_start = padding.top == 0 ? 0 : (padding.top - 1) / 2 + 1;
            int h_end = padding.bottom == 0 ? output_height : output_height - ((padding.bottom - 1) / 2 + 1);
            int w_start = padding.left == 0 ? 0 : (padding.left - 1) / 2 + 1;
            int w_end = padding.right == 0 ? output_width : output_width - ((padding.right - 1) / 2 + 1);

            const float* pinput = x.data<float>();
            const float* pkernel = weight.data<float>();
            float *poutput = out.data<float>();

            for (int n = 0; n < input_shape[0]; n++) {
#ifdef TS_USE_OPENMP
#ifdef TS_ON_ARMV7
#else
#pragma omp parallel for num_threads(openmp_threads()) 
#endif
#endif
                for (int c = 0; c < output_shape[1]; c++) {
                    const float* input_at = pinput + n * input_num_offset + c * input_channel_offset;
                    const float* kernel_at = pkernel + c * 9;

                    float* out_at = poutput + n * output_num_offset + c * output_channel_offset;

                    float32x4 k0_x4(kernel_at);
                    float32x4 k1_x4(kernel_at + 3);
                    float32x4 k2_x4(kernel_at + 6);

                    int h, w;

                    for (h = 0; h < h_start; h++) {
                        for (w = 0; w < output_width; w++){
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }

                    for (h = h_start; h < h_end; h++) {
                        for (w = 0; w < w_start; w++) {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }

                        for (w = w_start; w + 3 < w_end; w += 4) {
                            float* out0 = out_at + h * output_width + w;
                            float32x4 out0_x4(0.f);

                            const float* i0 = input_at + (h * stride.height - padding.top) * input_width + (w * stride.width - padding.left);
                            const float* i1 = i0 + input_width;
                            const float* i2 = i1 + input_width;
                            // const float* i3 = i2 + input_width;
                            // {[i0,i0+2,i0+4,i0+6],[i0+1,i0+3,i0+5,i0+7],[i0+2,i0+4,i0+6,i0+8]}
                            float32x4 i00_x4 = inc_load(i0, 2), i01_x4 = inc_load((i0 + 1), 2), i02_x4 = inc_load((i0 + 2), 2);
                            float32x4 i10_x4 = inc_load(i1, 2), i11_x4 = inc_load((i1 + 1), 2), i12_x4 = inc_load((i1 + 2), 2);
                            float32x4 i20_x4 = inc_load(i2, 2), i21_x4 = inc_load((i2 + 1), 2), i22_x4 = inc_load((i2 + 2), 2);

                            out0_x4 = fmadd(i00_x4, k0_x4, out0_x4, 0);
                            out0_x4 = fmadd(i01_x4, k0_x4, out0_x4, 1);
                            out0_x4 = fmadd(i02_x4, k0_x4, out0_x4, 2);
                            out0_x4 = fmadd(i10_x4, k1_x4, out0_x4, 0);
                            out0_x4 = fmadd(i11_x4, k1_x4, out0_x4, 1);
                            out0_x4 = fmadd(i12_x4, k1_x4, out0_x4, 2);
                            out0_x4 = fmadd(i20_x4, k2_x4, out0_x4, 0);
                            out0_x4 = fmadd(i21_x4, k2_x4, out0_x4, 1);
                            out0_x4 = fmadd(i22_x4, k2_x4, out0_x4, 2);

                            out0_x4.store(out0);
                        }

                        for (; w < output_width; w++) {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }
                    for (; h < output_height; h++) {
                        for (w = 0; w < output_width; w++) {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }
                }
            }
        }


        template<typename T>
        static void depthwise_3x3_s2_inplace(
            const Tensor &x,
            const Padding2D &padding,
            float padding_value,
            const Tensor &weight,
            const Stride2D &stride,
            const Dilation2D &dilation,
            Tensor &out) {

            auto weight_shape = weight.sizes();
            auto output_shape = out.sizes();
            auto input_shape = x.sizes();
            int input_height = input_shape[2];
            int input_width = input_shape[3];
            int output_height = output_shape[2];
            int output_width = output_shape[3];

            int input_channel_offset = input_shape[2] * input_shape[3];
            int input_num_offset = input_shape[1] * input_channel_offset;
            int output_channel_offset = output_shape[2] * output_shape[3];
            int output_num_offset = output_shape[1] * output_channel_offset;

            //h_start = padding.top == 0 ? 0 : (padding.top - 1) / stride.height + 1;
            //h_end = padding.bottom == 0 ? output_height : output_height - ((padding.bottom - 1) / stride.height + 1)
            //w_strst = padding.left == 0 ? 0 : (padding.left - 1) / stride.width + 1
            //w_end = padding.right == 0 ? output_width : output_width - ((padding.right - 1) / stride.width + 1)
            int h_start = padding.top == 0 ? 0 : padding.top;
            int h_end = padding.bottom == 0 ? output_height : output_height - padding.bottom;
            int w_start = padding.left == 0 ? 0 : padding.left;
            int w_end = padding.right == 0 ? output_width : output_width - padding.right;

            const float* pinput = x.data<float>();
            const float* pkernel = weight.data<float>();
            float *poutput = out.data<float>();

            for (int n = 0; n < input_shape[0]; n++) {
#ifdef TS_USE_OPENMP
#ifdef TS_ON_ARMV7
#else
#pragma omp parallel for num_threads(openmp_threads()) 
#endif
#endif
                for (int c = 0; c < output_shape[1]; c++) {
                    const float* input_at = pinput + n * input_num_offset + c * input_channel_offset;
                    const float* kernel_at = pkernel + c * 9;

                    float* out_at = poutput + n * output_num_offset + c * output_channel_offset;
                    //#ifdef TS_USE_AVX
                    //                    float32x4x2 k0_x4(kernel_at);
                    //                    float32x4x2 k1_x4(kernel_at + 1);
                    //#else
                    float32x4 k0_x4(kernel_at);
                    float32x4 k1_x4(kernel_at + 3);
                    float32x4 k2_x4(kernel_at + 6);
                    //#endif

                    int h, w;

                    for (h = 0; h < h_start; h++) {
                        for (w = 0; w < output_width; w++)
                        {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }

                    for (h = h_start; h + 1< h_end; h += 2) {
                        for (w = 0; w < w_start; w++) {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);

                            depthwise_inner(input_at, kernel_at,
                                h + 1, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }

                        for (w = w_start; w + 3 < w_end; w += 4) {
                            float* out0 = out_at + h * output_width + w;
                            float* out1 = out0 + output_width;
                            float32x4 out0_x4(0.f),out1_x4(0.f);

                            const float* i0 = input_at + (h * stride.height - padding.top) * input_width + (w * stride.width - padding.left);
                            const float* i1 = i0 + input_width;
                            const float* i2 = i1 + input_width;
                            const float* i3 = i2 + input_width;

                            //{[i0,i0+2,i0+4,i0+6],[i0+1,i0+3,i0+5,i0+7],[i0+2,i0+4,i0+6,i0+8]}
                            float32x4 i00_x4 = inc_load(i0, 2), i01_x4 = inc_load((i0 + 1), 2), i02_x4 = inc_load((i0 + 2), 2);
                            float32x4 i10_x4 = inc_load(i1, 2), i11_x4 = inc_load((i1 + 1), 2), i12_x4 = inc_load((i1 + 2), 2);
                            float32x4 i20_x4 = inc_load(i2, 2), i21_x4 = inc_load((i2 + 1), 2), i22_x4 = inc_load((i2 + 2), 2);
                            float32x4 i30_x4 = inc_load(i3, 2), i31_x4 = inc_load((i3 + 1), 2), i32_x4 = inc_load((i3 + 2), 2);

                            out0_x4 = fmadd(i00_x4, k0_x4, out0_x4, 0);
                            out0_x4 = fmadd(i01_x4, k0_x4, out0_x4, 1);
                            out0_x4 = fmadd(i02_x4, k0_x4, out0_x4, 2);
                            out0_x4 = fmadd(i10_x4, k1_x4, out0_x4, 0);
                            out0_x4 = fmadd(i11_x4, k1_x4, out0_x4, 1);
                            out0_x4 = fmadd(i12_x4, k1_x4, out0_x4, 2);
                            out0_x4 = fmadd(i20_x4, k2_x4, out0_x4, 0);
                            out0_x4 = fmadd(i21_x4, k2_x4, out0_x4, 1);
                            out0_x4 = fmadd(i22_x4, k2_x4, out0_x4, 2);

                            out1_x4 = fmadd(i10_x4, k0_x4, out1_x4, 0);
                            out1_x4 = fmadd(i11_x4, k0_x4, out1_x4, 1);
                            out1_x4 = fmadd(i12_x4, k0_x4, out1_x4, 2);
                            out1_x4 = fmadd(i20_x4, k1_x4, out1_x4, 0);
                            out1_x4 = fmadd(i21_x4, k1_x4, out1_x4, 1);
                            out1_x4 = fmadd(i22_x4, k1_x4, out1_x4, 2);
                            out1_x4 = fmadd(i30_x4, k2_x4, out1_x4, 0);
                            out1_x4 = fmadd(i31_x4, k2_x4, out1_x4, 1);
                            out1_x4 = fmadd(i32_x4, k2_x4, out1_x4, 2);

                            out0_x4.store(out0); out1_x4.store(out1);
                        }

                        for (; w < output_width; w++) {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);

                            depthwise_inner(input_at, kernel_at,
                                h + 1, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }
                    for (; h < output_height; h++) {
                        for (w = 0; w < output_width; w++) {
                            depthwise_inner(input_at, kernel_at,
                                h, w,
                                padding, padding_value, stride,
                                input_height, input_width,
                                output_width, out_at);
                        }
                    }
                }
            }
        }
    }
}

template class ts::cpu::DepthwiseConv2dAlgorithm<ts::dtype<ts::FLOAT32>::declare>;
template class ts::cpu::DepthwiseConv2dAlgorithm<ts::dtype<ts::FLOAT64>::declare>;