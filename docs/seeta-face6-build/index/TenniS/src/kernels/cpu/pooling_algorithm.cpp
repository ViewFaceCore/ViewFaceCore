//
// Created by yang on 2019/9/11.
//

#include <kernels/cpu/pooling_algorithm.h>
#include <kernels/common/simd.h>
#include <algorithm>

namespace ts{
    namespace cpu{

        template<typename T>
        void PoolingAlgorithm<T>::max_pooling_k3s2(const Tensor &input,
                                                   Tensor &out,
                                                   const Padding2D &padding){

        }

        template<>
        void PoolingAlgorithm<float>::max_pooling_k3s2(const Tensor &input,
                                                       Tensor &out,
                                                       const Padding2D &padding){
            const float* input_ptr = input.data<float>();
            float* output_ptr = out.data<float>();

            auto input_shape = input.sizes();
            auto output_shape = out.sizes();

            int input_channel = input_shape[1];
            int input_height = input_shape[2];
            int input_width = input_shape[3];
            int output_channel = output_shape[1];
            int output_height = output_shape[2];
            int output_width = output_shape[3];
            int input_channel_offset = input_height * input_width;
            int in_num_offset = input_channel * input_channel_offset;
            int out_channel_offset = output_height * output_width;
            int out_num_offset = output_channel * out_channel_offset;

            int real_width = 2 * output_width + 1; //(output_width - 1) * stride.w + kernel.w;
            int real_height = 2 * output_height + 1; //(output_height - 1) * stride.h + kernel.h;
            int pad_w_all = real_width - input_width;
            int pad_h_all = real_height - input_height;
            if(pad_w_all < 0)
                pad_w_all = 0;
            if(pad_h_all < 0)
                pad_h_all = 0;
            int real_pad_right = pad_w_all - padding.left;
            int real_pad_bottom = pad_h_all - padding.top;

            output_height = real_pad_bottom > 0 ? output_height - 1 : output_height;
            output_width = real_pad_right > 0 ? output_width - 1 : output_width;

            int width_blocks = padding.left == 1 ? (output_width - 1) / 4 : output_width / 4;
            int input_w_remain = padding.left == 1 ? input_width - output_width * 2 + 1 : input_width - output_width * 2;

            for (int n = 0; n < input_shape[0]; ++n) {
                for (int c = 0; c < input_channel; ++c) {
                    const float* input_at = input_ptr + n * in_num_offset + c * input_channel_offset;
                    const float* i0 = input_at;
                    const float* i1 = i0 + input_width;
                    const float* i2 = i1 + input_width;
                    float* out_at = output_ptr + n * out_num_offset + c * out_channel_offset;

                    //top
                    int h = 0;
                    if(padding.top == 1) {
                        h++;
                        if (padding.left == 1) {
                            *out_at = std::max(std::max(i0[0], i0[1]), std::max(i1[0], i1[1]));
                            out_at++;
                            i0++;
                            i1++;
                        }
#ifdef TS_USE_NEON
                        float32x4x2 i0_x4x2 = incx4x2_load(i0, 2);
                        float32x4x2 i1_x4x2 = incx4x2_load(i1, 2);
#endif
                        for (int w = 0; w < width_blocks; ++w) {
#ifdef TS_USE_NEON
                            float32x4x2 i01_x4x2_temp = incx4x2_load(i0 + 8, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_x4x2[0], i0_x4x2[1]);
                            float32x4 i01_x4x2 = concat(i0_x4x2[0], i01_x4x2_temp[0], 1);
                            i0_max_x4 = max_float32x4(i0_max_x4, i01_x4x2);

                            float32x4x2 i11_x4x2_temp = incx4x2_load(i1 + 8, 2);
                            float32x4 i1_max_x4 = max_float32x4(i1_x4x2[0], i1_x4x2[1]);
                            float32x4 i11_x4x2 = concat(i1_x4x2[0], i11_x4x2_temp[0], 1);
                            i1_max_x4 = max_float32x4(i1_max_x4, i11_x4x2);

                            i0_max_x4 = max_float32x4(i0_max_x4, i1_max_x4);

                            i0_x4x2 = i01_x4x2_temp;
                            i1_x4x2 = i11_x4x2_temp;
#else
                            float32x4 i0_1357 = inc_load(i0, 2);
                            float32x4 i0_2468 = inc_load(i0 + 1, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_1357, i0_2468);
                            float32x4 i0_3579 = inc_load(i0 + 2, 2);
                            i0_max_x4 = max_float32x4(i0_max_x4, i0_3579);

                            float32x4 i1_1357 = inc_load(i1, 2);
                            float32x4 i1_2468 = inc_load(i1 + 1, 2);
                            float32x4 i1_max_x4 = max_float32x4(i1_1357, i1_2468);
                            float32x4 i1_3579 = inc_load(i1 + 2, 2);
                            i1_max_x4 = max_float32x4(i1_max_x4, i1_3579);

                            i0_max_x4 = max_float32x4(i0_max_x4, i1_max_x4);
#endif
                            i0_max_x4.store(out_at);
                            i0 += 8;
                            i1 += 8;
                            out_at += 4;
                        }
                        int remain_w_index = padding.left == 1 ? width_blocks * 4 + 1 : width_blocks * 4;
                        for (int w = remain_w_index; w < output_width; ++w) {
                            float i0_max = std::max(std::max(i0[0], i0[1]), i0[2]);
                            float i1_max = std::max(std::max(i1[0], i1[1]), i1[2]);
                            *out_at = std::max(i0_max, i1_max);

                            i0 += 2;i1 += 2;
                            out_at++;
                        }
                        if (real_pad_right == 1) {
                            float i0_max = std::max(i0[0], i0[1]);
                            float i1_max = std::max(i1[0], i1[1]);
                            *out_at = std::max(i0_max, i1_max);
                            out_at++;
                        }
                        else if(real_pad_right == 2){
                            *out_at = std::max(i0[0], i1[0]);
                            out_at++;
                        }

                        i0 += input_w_remain;
                        i1 += input_w_remain;
                    }

                    //center
                    i2 = i1 + input_width;
                    for (; h < output_height; ++h) {
                        if(padding.left == 1){
                            float i0_max = std::max(i0[0], i0[1]);
                            float i1_max = std::max(i1[0], i1[1]);
                            float i2_max = std::max(i2[0], i2[1]);
                            *out_at = std::max(std::max(i0_max, i1_max), i2_max);
                            i0++;i1++;i2++;
                            out_at++;
                        }
#ifdef TS_USE_NEON
                        float32x4x2 i0_x4x2 = incx4x2_load(i0, 2);
                        float32x4x2 i1_x4x2 = incx4x2_load(i1, 2);
                        float32x4x2 i2_x4x2 = incx4x2_load(i2, 2);
#endif
                        for (int w = 0; w < width_blocks; ++w) {
#ifdef TS_USE_NEON
                            float32x4x2 i01_x4x2_temp = incx4x2_load(i0 + 8, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_x4x2[0], i0_x4x2[1]);
                            float32x4 i01_x4x2 = concat(i0_x4x2[0], i01_x4x2_temp[0], 1);
                            i0_max_x4 = max_float32x4(i0_max_x4, i01_x4x2);

                            float32x4x2 i11_x4x2_temp = incx4x2_load(i1 + 8, 2);
                            float32x4 i1_max_x4 = max_float32x4(i1_x4x2[0], i1_x4x2[1]);
                            float32x4 i11_x4x2 = concat(i1_x4x2[0], i11_x4x2_temp[0], 1);
                            i1_max_x4 = max_float32x4(i1_max_x4, i11_x4x2);

                            float32x4x2 i21_x4x2_temp = incx4x2_load(i2 + 8, 2);
                            float32x4 i2_max_x4 = max_float32x4(i2_x4x2[0], i2_x4x2[1]);
                            float32x4 i21_x4x2 = concat(i2_x4x2[0], i21_x4x2_temp[0], 1);
                            i2_max_x4 = max_float32x4(i2_max_x4, i21_x4x2);

                            i0_max_x4 = max_float32x4(max_float32x4(i0_max_x4, i1_max_x4), i2_max_x4);

                            i0_x4x2 = i01_x4x2_temp;
                            i1_x4x2 = i11_x4x2_temp;
                            i2_x4x2 = i21_x4x2_temp;
#else
                            float32x4 i0_1357 = inc_load(i0, 2);
                            float32x4 i0_2468 = inc_load(i0 + 1, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_1357, i0_2468);
                            float32x4 i0_3579 = inc_load(i0 + 2, 2);
                            i0_max_x4 = max_float32x4(i0_max_x4, i0_3579);

                            float32x4 i1_1357 = inc_load(i1, 2);
                            float32x4 i1_2468 = inc_load(i1 + 1, 2);
                            float32x4 i1_max_x4 = max_float32x4(i1_1357, i1_2468);
                            float32x4 i1_3579 = inc_load(i1 + 2, 2);
                            i1_max_x4 = max_float32x4(i1_max_x4, i1_3579);

                            float32x4 i2_1357 = inc_load(i2, 2);
                            float32x4 i2_2468 = inc_load(i2 + 1, 2);
                            float32x4 i2_max_x4 = max_float32x4(i2_1357, i2_2468);
                            float32x4 i2_3579 = inc_load(i2 + 2, 2);
                            i2_max_x4 = max_float32x4(i2_max_x4, i2_3579);

                            i0_max_x4 = max_float32x4(max_float32x4(i0_max_x4, i1_max_x4), i2_max_x4);
#endif
                            i0_max_x4.store(out_at);
                            i0 += 8;i1 += 8;i2 += 8;
                            out_at += 4;
                        }
                        int remain_w_index = padding.left == 1 ? width_blocks * 4 + 1 : width_blocks * 4;
                        for (int w = remain_w_index; w < output_width; ++w) {
                            float i0_max = std::max(std::max(i0[0], i0[1]), i0[2]);
                            float i1_max = std::max(std::max(i1[0], i1[1]), i1[2]);
                            float i2_max = std::max(std::max(i2[0], i2[1]), i2[2]);
                            *out_at = std::max(std::max(i0_max, i1_max), i2_max);

                            i0 += 2;i1 += 2;i2 += 2;
                            out_at++;
                        }
                        if(real_pad_right == 1){
                            float i0_max = std::max(i0[0], i0[1]);
                            float i1_max = std::max(i1[0], i1[1]);
                            float i2_max = std::max(i2[0], i2[1]);
                            *out_at = std::max(std::max(i0_max, i1_max), i2_max);
                            out_at++;
                        }
                        else if(real_pad_right == 2){
                            *out_at = std::max(std::max(i0[0], i1[0]), i2[0]);
                            out_at++;
                        }

                        i0 += input_w_remain + input_width;
                        i1 += input_w_remain + input_width;
                        i2 += input_w_remain + input_width;
                    }

                    //bottom
                    if(real_pad_bottom == 1){
                        if(padding.left == 1){
                            float i0_max = std::max(i0[0], i0[1]);
                            float i1_max = std::max(i1[0], i1[1]);
                            *out_at = std::max(i0_max, i1_max);
                            i0++;i1++;
                            out_at++;
                        }
#ifdef TS_USE_NEON
                        float32x4x2 i0_x4x2 = incx4x2_load(i0, 2);
                        float32x4x2 i1_x4x2 = incx4x2_load(i1, 2);
#endif
                        for (int w = 0; w < width_blocks; ++w) {
#ifdef TS_USE_NEON
                            float32x4x2 i01_x4x2_temp = incx4x2_load(i0 + 8, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_x4x2[0], i0_x4x2[1]);
                            float32x4 i01_x4x2 = concat(i0_x4x2[0], i01_x4x2_temp[0], 1);
                            i0_max_x4 = max_float32x4(i0_max_x4, i01_x4x2);

                            float32x4x2 i11_x4x2_temp = incx4x2_load(i1 + 8, 2);
                            float32x4 i1_max_x4 = max_float32x4(i1_x4x2[0], i1_x4x2[1]);
                            float32x4 i11_x4x2 = concat(i1_x4x2[0], i11_x4x2_temp[0], 1);
                            i1_max_x4 = max_float32x4(i1_max_x4, i11_x4x2);

                            i0_max_x4 = max_float32x4(i0_max_x4, i1_max_x4);

                            i0_x4x2 = i01_x4x2_temp;
                            i1_x4x2 = i11_x4x2_temp;
#else
                            float32x4 i0_1357 = inc_load(i0, 2);
                            float32x4 i0_2468 = inc_load(i0 + 1, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_1357, i0_2468);
                            float32x4 i0_3579 = inc_load(i0 + 2, 2);
                            i0_max_x4 = max_float32x4(i0_max_x4, i0_3579);

                            float32x4 i1_1357 = inc_load(i1, 2);
                            float32x4 i1_2468 = inc_load(i1 + 1, 2);
                            float32x4 i1_max_x4 = max_float32x4(i1_1357, i1_2468);
                            float32x4 i1_3579 = inc_load(i1 + 2, 2);
                            i1_max_x4 = max_float32x4(i1_max_x4, i1_3579);

                            i0_max_x4 = max_float32x4(i0_max_x4, i1_max_x4);
#endif
                            i0_max_x4.store(out_at);
                            i0 += 8;i1 += 8;
                            out_at += 4;
                        }
                        int remain_w_index = padding.left == 1 ? width_blocks * 4 + 1 : width_blocks * 4;
                        for (int w = remain_w_index; w < output_width; ++w) {
                            float i0_max = std::max(std::max(i0[0], i0[1]), i0[2]);
                            float i1_max = std::max(std::max(i1[0], i1[1]), i1[2]);
                            *out_at = std::max(i0_max, i1_max);
                            i0 += 2;i1 += 2;
                            out_at++;
                        }
                        if(real_pad_right == 1){
                            float i0_max = std::max(i0[0], i0[1]);
                            float i1_max = std::max(i1[0], i1[1]);
                            *out_at = std::max(i0_max,i1_max);
                            out_at++;
                        }
                        else if(real_pad_right == 2){
                            *out_at = std::max(i0[0], i1[0]);
                            out_at++;
                        }
                    }
                    else if(real_pad_bottom == 2){
                        if(padding.left == 1){
                            *out_at = std::max(i0[0], i0[1]);
                            i0++;
                            out_at++;
                        }
#ifdef TS_USE_NEON
                        float32x4x2 i0_x4x2 = incx4x2_load(i0, 2);
#endif
                        for (int w = 0; w < width_blocks; ++w) {
#ifdef TS_USE_NEON

                            float32x4x2 i01_x4x2_temp = incx4x2_load(i0 + 8, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_x4x2[0], i0_x4x2[1]);
                            float32x4 i01_x4x2 = concat(i0_x4x2[0], i01_x4x2_temp[0], 1);
                            i0_max_x4 = max_float32x4(i0_max_x4, i01_x4x2);
                            i0_x4x2 = i01_x4x2_temp;
#else
                            float32x4 i0_1357 = inc_load(i0, 2);
                            float32x4 i0_2468 = inc_load(i0 + 1, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_1357, i0_2468);
                            float32x4 i0_3579 = inc_load(i0 + 2, 2);
                            i0_max_x4 = max_float32x4(i0_max_x4, i0_3579);
#endif
                            i0_max_x4.store(out_at);
                            i0 += 8;
                            out_at += 4;
                        }
                        int remain_w_index = padding.left == 1 ? width_blocks * 4 + 1 : width_blocks * 4;
                        for (int w = remain_w_index; w < output_width; ++w) {
                            float i0_max = std::max(std::max(i0[0], i0[1]), i0[2]);
                            *out_at = i0_max;
                            i0 += 2;
                            out_at++;
                        }
                        if(real_pad_right == 1){
                            *out_at = std::max(i0[0], i0[1]);
                            out_at++;
                        }
                        else if(real_pad_right == 2){
                            *out_at = i0[0];
                            out_at++;
                        }
                    }
                }
            }
        }

        template<typename T>
        void PoolingAlgorithm<T>::max_pooling_k2s2(const Tensor &input,
                                                   Tensor &out,
                                                   const Padding2D &padding){

        }

        template<>
        void PoolingAlgorithm<float>::max_pooling_k2s2(const Tensor &input,
                                                       Tensor &out,
                                                       const Padding2D &padding){
            const float* input_ptr = input.data<float>();
            float* output_ptr = out.data<float>();

            auto input_shape = input.sizes();
            auto output_shape = out.sizes();

            int input_channel = input_shape[1];
            int input_height = input_shape[2];
            int input_width = input_shape[3];
            int output_channel = output_shape[1];
            int output_height = output_shape[2];
            int output_width = output_shape[3];
            int input_channel_offset = input_height * input_width;
            int in_num_offset = input_channel * input_channel_offset;
            int out_channel_offset = output_height * output_width;
            int out_num_offset = output_channel * out_channel_offset;

            int real_width = 2 * output_width; //(output_width - 1) * stride.w + kernel.w;
            int real_height = 2 * output_height; //(output_height - 1) * stride.h + kernel.h;
            int pad_w_all = real_width - input_width;
            int pad_h_all = real_height - input_height;
            if(pad_w_all < 0)
                pad_w_all = 0;
            if(pad_h_all < 0)
                pad_h_all = 0;
            int real_pad_right = pad_w_all - padding.left;
            int real_pad_bottom = pad_h_all - padding.top;

            output_height = real_pad_bottom > 0 ? output_height - 1 : output_height;
            output_width = real_pad_right > 0 ? output_width - 1 : output_width;

            int width_blocks = padding.left == 1 ? (output_width - 1) / 4 : output_width / 4;
            int input_w_remain = padding.left == 1 ? input_width - output_width * 2 + 1 : input_width - output_width * 2;
            for (int n = 0; n < input_shape[0]; ++n) {
                for (int c = 0; c < input_channel; ++c) {
                    const float* input_at = input_ptr + n * in_num_offset + c * input_channel_offset;
                    const float* i0 = input_at;
                    const float* i1 = i0 + input_width;
                    float* out_at = output_ptr + n * out_num_offset + c * out_channel_offset;

                    //top
                    int h = 0;
                    if(padding.top == 1) {
                        h++;
                        if(padding.left == 1){
                            *out_at = i0[0];
                            i0++;
                            out_at++;
                        }
                        for (int w = 0; w < width_blocks; ++w) {
                            float32x4 i0_1357 = inc_load(i0, 2);
                            float32x4 i0_2468 = inc_load(i0 + 1, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_1357, i0_2468);
                            i0_max_x4.store(out_at);

                            i0 += 8;
                            out_at += 4;
                        }
                        int remain_w_index = padding.left == 1 ? width_blocks * 4 + 1 : width_blocks * 4;
                        for (int w = remain_w_index; w < output_width; ++w) {
                            *out_at = std::max(i0[0], i0[1]);
                            i0 += 2;
                            out_at++;
                        }
                        if(real_pad_right > 0){
                            *out_at = i0[0];
                            out_at++;
                        }
                        i0 += input_w_remain;
                    }

                    //center
                    i1 = i0 + input_width;
                    for (; h < output_height; ++h) {
                        if(padding.left == 1){
                            *out_at = std::max(i0[0], i1[0]);
                            i0++;i1++;
                            out_at++;
                        }
                        for (int w = 0; w < width_blocks; ++w) {
                            float32x4 i0_1357 = inc_load(i0, 2);
                            float32x4 i0_2468 = inc_load(i0 + 1, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_1357, i0_2468);

                            float32x4 i1_1357 = inc_load(i1, 2);
                            float32x4 i1_2468 = inc_load(i1 + 1, 2);
                            float32x4 i1_max_x4 = max_float32x4(i1_1357, i1_2468);

                            i0_max_x4 = max_float32x4(i0_max_x4, i1_max_x4);
                            i0_max_x4.store(out_at);

                            i0 += 8;i1 += 8;
                            out_at += 4;
                        }
                        int remain_w_index = padding.left == 1 ? width_blocks * 4 + 1 : width_blocks * 4;
                        for (int w = remain_w_index; w < output_width; ++w) {
                            float i0_max = std::max(i0[0], i0[1]);
                            float i1_max = std::max(i1[0], i1[1]);
                            *out_at = std::max(i0_max, i1_max);
                            i0 += 2; i1 += 2;
                            out_at++;
                        }
                        if(real_pad_right > 0){
                            *out_at = std::max(i0[0], i1[0]);
                            out_at++;
                        }
                        i0 += input_w_remain + input_width;
                        i1 += input_w_remain + input_width;
                    }

                    //bottom
                    if(real_pad_bottom > 0){
                        if(padding.left == 1){
                            *out_at = i0[0];
                            i0++;
                            out_at++;
                        }
                        for (int w = 0; w < width_blocks; ++w) {
                            float32x4 i0_1357 = inc_load(i0, 2);
                            float32x4 i0_2468 = inc_load(i0 + 1, 2);
                            float32x4 i0_max_x4 = max_float32x4(i0_1357, i0_2468);
                            i0_max_x4.store(out_at);

                            i0 += 8;
                            out_at += 4;
                        }
                        int remain_w_index = padding.left == 1 ? width_blocks * 4 + 1 : width_blocks * 4;
                        for (int w = remain_w_index; w < output_width; ++w) {
                            *out_at = std::max(i0[0], i0[1]);
                            i0 += 2;
                            out_at++;
                        }
                        if(real_pad_right > 0){
                            *out_at = i0[0];
                            out_at++;
                        }
                    }
                }
            }
        }

    }//cpu
}//ts

template class ts::cpu::PoolingAlgorithm<ts::dtype<ts::FLOAT32>::declare>;
template class ts::cpu::PoolingAlgorithm<ts::dtype<ts::FLOAT64>::declare>;
