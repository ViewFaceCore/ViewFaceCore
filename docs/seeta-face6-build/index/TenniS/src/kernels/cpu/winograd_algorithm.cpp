//
// Created by yang on 2019/10/21.
//

#include <backend/common_structure.h>
#include "kernels/cpu/winograd_algorithm.h"
#include "kernels/common/function.h"
#include "kernels/cpu/math_cpu.h"
#include "kernels/cpu/pad2d_algorithm.h"
#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif
#include <array>

namespace ts{
    namespace cpu{

        /*
         * kernel =
         * [
         *     [K0c_0[0,.....9]],
         *     [K0c_1[0,.....9]],
         *          ...
         *     [K0c_M[0,.....9]],
         *          ...
         *     [KNc_M[0,.....9]],
         *
         * ],N for kernel num
         * ===============================>
         * U = G(kernel)Gt
         * ===============================>
         * kernel_trans = pack U:OcIcHW->TOcIc(T for tile count)
         * [                          |
         *     [U0[C_0,C_1,.....C_M], |
         *     [U1[C_0,C_1,.....C_M], |
         *                     ...    |  x 16
         *     [UN[C_0,C_1,.....C_M]  |
         * ]                          |
         * ===============================>
         * kernel_tm = gemm pack A(kernel_trans)
         *
         * G =
         * [
         *     [1.0f,  0.0f,  0.0f],
         *     [0.5f,  0.5f,  0.5f],
         *     [0.5f, -0.5f,  0.5f],
         *     [0.0f,  0.0f,  1.0f]
         * ]
         *
         */
        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f23_transform_and_pack_kernel(const Tensor& kernel, int in_tile_size, Tensor &kernel_tm){

        }

        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f23_transform_and_pack_kernel(const Tensor& kernel, int in_tile_size, Tensor &kernel_tm){
            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];
            int stride = out_channel * input_channel;

            Tensor kernel_trans(Tensor::InFlow::HOST, kernel_tm.dtype(), kernel_tm.sizes());

            const float *p_kernel = kernel.data<float>();
            int kernel_num_offset = input_channel * 9;
            float *p_kernel_trans = kernel_trans.data<float>();

            const float G[4][3] = {
                { 1.f,     0.f,     0.f },
                { 1.f / 2,   1.f / 2,   1.f / 2 },
                { 1.f / 2,   -1.f / 2,   1.f / 2 },
                { 0.f,     0.f,     1.f }
            };

            for (int p = 0; p < out_channel; p++)
            {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int q = 0; q < input_channel; q++)
                {
                    const float *kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    float *kernel_trans_at = p_kernel_trans + p * input_channel + q;

                    // transform kernel
                    const float *k0 = kernel_at;
                    const float *k1 = kernel_at + 3;
                    const float *k2 = kernel_at + 6;

                    float tmp[3][4];
                    for (int i = 0; i < 4; i++)
                    {
                        tmp[0][i] = k0[0] * G[i][0] + k0[1] * G[i][1] + k0[2] * G[i][2];
                        tmp[1][i] = k1[0] * G[i][0] + k1[1] * G[i][1] + k1[2] * G[i][2];
                        tmp[2][i] = k2[0] * G[i][0] + k2[1] * G[i][1] + k2[2] * G[i][2];
                    }

                    // U,pack:OcIcHW->TOcIc(T for tile count)
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            kernel_trans_at[(i * 4 + j) * stride] = tmp[0][j] * G[i][0] + tmp[1][j] * G[i][1] + tmp[2][j] * G[i][2];
                        }
                    }
                }
            }

            //gemm pack A
            float *kernel_tm_ptr = kernel_tm.data<float>();
            const float* from = p_kernel_trans;
            float *to = kernel_tm_ptr;
            int transform_kernel_tile_offset = out_channel * input_channel;
            for (int i = 0; i < in_tile_size; ++i) {
                math<float,float>::pack8_A(out_channel, input_channel, from, input_channel, to);
                from += transform_kernel_tile_offset;
                to += transform_kernel_tile_offset;
            }


        };

        /*
         * x =
         * [
         *     [Ic_0[0,.....N]],
         *     [Ic_1[0,.....N]],
         *          ...
         *     [Ic_M[0,.....N]],
         * ]
         * ===============================>
         * V = BT(x)B
         * ===============================>
         * x_tm = pack V:NCHW->NTCB(T for tile count,B for tile indices)
         * [                                              |
         *     [Vc_0[tile0_0......tile(tile_count-1)_0]], |
         *     [Vc_1[tile0_0......tile(tile_count-1)_0]], |
         *                     ...                        |  x 16
         *     [Vc_M[tile0_0......tile(tile_count-1)_0]]  |
         * ]                                              |
         *
         * BT =
         * [
         *     [1.0f,  0.0f, -1.0f,  0.0f],
         *     [0.0f,  1.0f,  1.00f, 0.0f],
         *     [0.0f, -1.0f,  1.00f, 0.0f],
         *     [0.0f, -1.0f,  0.00f, 1.0f]
         * ]
         *
         */
        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f23_transform_and_pack_input(const Tensor& x, int tile_count, Tensor &x_tm){

        }

        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f23_transform_and_pack_input(const Tensor& x, int tile_count, Tensor &x_tm){
            Shape input_shape = x.sizes();
            int num = input_shape[0];
            int input_channel = input_shape[1];
            int input_height = input_shape[2];
            int input_width = input_shape[3];
            int input_channel_offset = input_height * input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int stride = input_channel * tile_count;
            int out_num_offset = stride * 16;

            const float *input_ptr = x.data<float>();
            float *out_ptr = x_tm.data<float>();

            for (int n = 0; n < num; ++n) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; ++c) {

                    float t00,t01,t02,t03;
                    float t10,t11,t12,t13;
                    float t20,t21,t22,t23;
                    float t30,t31,t32,t33;

                    int tile_index = 0;
                    const float *input_cur = input_ptr + n * input_num_offset + c * input_channel_offset;
                    float *out_cur = out_ptr + n * out_num_offset + c * tile_count;
                    for (int h = 0; h + 2 < input_height; h += 2) {
                        for (int w = 0; w + 2 < input_width; w += 2) {
                            const float *input_at = input_cur + h * input_width + w;
                            const float *i0 = input_at;
                            const float *i1 = i0 + input_width;
                            const float *i2 = i1 + input_width;
                            const float *i3 = i2 + input_width;

                            float *out_at = out_cur + tile_index;

                            // t = BT * d * B
                            t00 = (i0[0] - i2[0]) - (i0[2] - i2[2]);
                            t01 = (i0[1] - i2[1]) + (i0[2] - i2[2]);
                            t02 = (i0[2] - i2[2]) - (i0[1] - i2[1]);
                            t03 = (i0[1] - i2[1]) - (i0[3] - i2[3]);

                            t10 = (i1[0] + i2[0]) - (i1[2] + i2[2]);
                            t11 = (i1[1] + i2[1]) + (i1[2] + i2[2]);
                            t12 = (i1[2] + i2[2]) - (i1[1] + i2[1]);
                            t13 = (i1[1] + i2[1]) - (i1[3] + i2[3]);

                            t20 = (i2[0] - i1[0]) - (i2[2] - i1[2]);
                            t21 = (i2[1] - i1[1]) + (i2[2] - i1[2]);
                            t22 = (i2[2] - i1[2]) - (i2[1] - i1[1]);
                            t23 = (i2[1] - i1[1]) - (i2[3] - i1[3]);

                            t30 = (i1[0] - i3[0]) - (i1[2] - i3[2]);
                            t31 = (i1[1] - i3[1]) + (i1[2] - i3[2]);
                            t32 = (i1[2] - i3[2]) - (i1[1] - i3[1]);
                            t33 = (i1[1] - i3[1]) - (i1[3] - i3[3]);

                            //pack:NCHW->NTCB(T for tile count,B for tile indices)
                            out_at[0] = t00;
                            out_at[1 * stride] = t01;
                            out_at[2 * stride] = t02;
                            out_at[3 * stride] = t03;

                            out_at[4 * stride] = t10;
                            out_at[5 * stride] = t11;
                            out_at[6 * stride] = t12;
                            out_at[7 * stride] = t13;

                            out_at[8 * stride] = t20;
                            out_at[9 * stride] = t21;
                            out_at[10 * stride] = t22;
                            out_at[11 * stride] = t23;

                            out_at[12 * stride] = t30;
                            out_at[13 * stride] = t31;
                            out_at[14 * stride] = t32;
                            out_at[15 * stride] = t33;

                            ++tile_index;
                        }
                    }
                }
            }
        }

        /* out_tm =
         * [                                           |
         *     Out_c_0[tile0_0,tile1_0,....tileN_0],   |
         *     Out_c_1[tile0_0,tile1_0,....tileN_0],   |
         *                  ...                        | x 16
         *     Out_c_M[tile0_0,tile1_0,....tileN_0],   |
         * ]                                           |
         *     N for tile_num
         * =======================================>
         * tmp
         * [
         *     Out_c_0[tile0_0,tile0_1,....tile0_16],
         *     Out_c_0[tile1_0,tile1_1,....tile1_16],
         *                  ...
         *     Out_c_0[tileN_0,tileN_1,....tileN_16],
         *                  ...
         *     Out_c_M[tileN_0,tileN_1,....tileN_16],
         * ]
         * ========================================>
         * out = AT(tmp)A
         * AT =
         * [
         *     [1.0f,1.0f,1.0f,0.0f],
         *     [0.0f,1.0f,-1.0f,1.0f]
         * ]
         */
        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f23_transform_output(const Tensor& out_tm, int tile_count, Tensor& out){

        }

        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f23_transform_output(const Tensor& out_tm, int tile_count, Tensor& out){
            Shape out_shape = out.sizes();
            Shape out_tm_shape = out_tm.sizes();
            int num = out_tm_shape[0];
            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];
            int stride = out_channel * tile_count;
            int out_tm_num_offset = 16 * stride;
            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const float* out_tm_ptr = out_tm.data<float>();
            float *out_ptr = out.data<float>();

            for (int n = 0; n < num; ++n) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < out_channel; ++c) {
                    int tile_offset = 0;
                    const float *out_tm_cur = out_tm_ptr + n * out_tm_num_offset + c * tile_count;
                    float *out_cur = out_ptr + n * out_num_offset + c * out_channel_offset;
                    for (int h = 0; h + 1 < out_height; h += 2) {
                        for (int w = 0; w + 1< out_width; w += 2) {
                            const float *out_tm_at = out_tm_cur + tile_offset;
                            float *out_at = out_cur + h * out_width + w;

                            float tmp[16];
                            tmp[0] = out_tm_at[0];
                            tmp[1] = out_tm_at[1 * stride];
                            tmp[2] = out_tm_at[2 * stride];
                            tmp[3] = out_tm_at[3 * stride];
                            tmp[4] = out_tm_at[4 * stride];
                            tmp[5] = out_tm_at[5 * stride];
                            tmp[6] = out_tm_at[6 * stride];
                            tmp[7] = out_tm_at[7 * stride];
                            tmp[8] = out_tm_at[8 * stride];
                            tmp[9] = out_tm_at[9 * stride];
                            tmp[10] = out_tm_at[10 * stride];
                            tmp[11] = out_tm_at[11 * stride];
                            tmp[12] = out_tm_at[12 * stride];
                            tmp[13] = out_tm_at[13 * stride];
                            tmp[14] = out_tm_at[14 * stride];
                            tmp[15] = out_tm_at[15 * stride];

                            float tmpA[8];
                            tmpA[0] = tmp[0] + tmp[1] + tmp[2];
                            tmpA[1] = tmp[1] - tmp[2] - tmp[3];
                            tmpA[2] = tmp[4] + tmp[5] + tmp[6];
                            tmpA[3] = tmp[5] - tmp[6] - tmp[7];
                            tmpA[4] = tmp[8] + tmp[9] + tmp[10];
                            tmpA[5] = tmp[9] - tmp[10] - tmp[11];
                            tmpA[6] = tmp[12] + tmp[13] + tmp[14];
                            tmpA[7] = tmp[13] - tmp[14] - tmp[15];

                            out_at[0] = tmpA[0] + tmpA[2] + tmpA[4];
                            out_at[1] = tmpA[1] + tmpA[3] + tmpA[5];
                            out_at[out_width] = tmpA[2] - tmpA[4] - tmpA[6];
                            out_at[out_width + 1] = tmpA[3] - tmpA[5] - tmpA[7];

                            ++tile_offset;
                        }
                    }
                }
            }
        }

        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f23(const Tensor &x,
                                             const Padding2D &padding,
                                             float padding_value,
                                             const Tensor &kernel,
                                             Tensor &out,
                                             bool kernel_transformed){

        }

        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f23(const Tensor &x,
                                                 const Padding2D &padding,
                                                 float padding_value,
                                                 const Tensor &kernel,
                                                 Tensor &out,
                                                 bool kernel_transformed){

            int tile_width = 2;
            int tile_height = 2;

            Shape kernel_shape = kernel.sizes();
            Tensor input_padded = x;
            Tensor out_padded = out;
            bool out_padded_flag = false;
            Stride2D stride(1, 1);
            KSize2D ksize(3, 3);
            KernelCommonFunc<float>::in_out_pad_and_fix_size(x,
                                                             kernel_shape,
                                                             out,
                                                             tile_height,
                                                             tile_width,
                                                             padding,
                                                             padding_value,
                                                             stride,
                                                             ksize,
                                                             input_padded,
                                                             out_padded,
                                                             out_padded_flag);

            auto output_shape = out_padded.sizes();
            auto input_shape = input_padded.sizes();
            auto src_output_shape = out.sizes();

            int num = input_shape[0];
            int input_channel = input_shape[1];
            int out_channel = output_shape[1];
            int out_height = output_shape[2];
            int out_width = output_shape[3];

            int tile_block_height = out_height / tile_height;
            int tile_block_width = out_width / tile_width;
            int tile_block_num = tile_block_height * tile_block_width;
            int in_tile_size = (tile_width + 2) * (tile_height + 2);

            //transform kernel
            Tensor kernel_tm = kernel;
            if(!kernel_transformed){
                Shape kernel_tm_shape = {in_tile_size, kernel_shape[0], kernel_shape[1]};
                Tensor kernel_tmp(Tensor::InFlow::HOST, kernel.dtype(), kernel_tm_shape);
//                std::memset(kernel_tmp.data<float>(), 0, sizeof(float)*kernel_tmp.count());
                winograd_f23_transform_and_pack_kernel(kernel, in_tile_size, kernel_tmp);
                kernel_tm = kernel_tmp;
            }
            const float *kernel_ptr = kernel_tm.data<float>();

            //transform input
            Shape input_tm_shape = {num, in_tile_size, input_channel, tile_block_num};
            Tensor input_tm(Tensor::InFlow::HOST, input_padded.dtype(), input_tm_shape);
            winograd_f23_transform_and_pack_input(input_padded, tile_block_num, input_tm);
            const float *trans_input_ptr = input_tm.data<float>();

            //eltwise_gemm->gemm(O = U*V)
            Shape transform_out_shape = {num, in_tile_size, out_channel, tile_block_num};
            Tensor transform_out(Tensor::InFlow::HOST, out.dtype(), transform_out_shape);
            float *trans_out_ptr = transform_out.data<float>();

            int transform_kernel_tile_offset = out_channel * input_channel;
            int transform_input_tile_offset = input_channel * tile_block_num;
            int transform_input_num_offset = in_tile_size * transform_input_tile_offset;
            int transform_out_tile_offset = out_channel * tile_block_num;
            int transform_out_num_offset = in_tile_size * transform_out_tile_offset;

            for (int n = 0; n < num; ++n) {
                const float *trans_input_cur = trans_input_ptr + n * transform_input_num_offset;
                float *trans_out_cur = trans_out_ptr + n * transform_out_num_offset;

                for (int i = 0; i < in_tile_size; ++i) {
                    const float *trans_kernel_at = kernel_ptr + i * transform_kernel_tile_offset;
                    const float *trans_input_at = trans_input_cur + i * transform_input_tile_offset;
                    float *trans_out_at = trans_out_cur + i * transform_out_tile_offset;
                    math<float, float>::gemm(out_channel,
                                             tile_block_num,
                                             input_channel,
                                             1.f,
                                             trans_kernel_at,
                                             trans_input_at,
                                             0.f,
                                             trans_out_at,
                                             false,
                                             true);
                }
            }

            //transform output
            winograd_f23_transform_output(transform_out, tile_block_num, out_padded);

            //cut output
            if (out_padded_flag) {
                std::array<int, 2> pad_h = {0, src_output_shape[2] - out_height};
                std::array<int, 2> pad_w = {0, src_output_shape[3] - out_width};
                PadAlgorithm<float>::cut2d(out_padded, pad_h, pad_w, 0.f, out);
            }
        }

        /*
         * kernel =
         * [
         *     [K0c_0[0,.....9]],
         *     [K0c_1[0,.....9]],
         *          ...
         *     [K0c_M[0,.....9]],
         *          ...
         *     [KNc_M[0,.....9]],
         *
         * ],N for kernel num
         * ===============================>
         * U = G(kernel)Gt
         * ===============================>
         * kernel_trans = pack U:OcIcHW->TOcIc(T for tile count)
         * [                          |
         *     [U0[C_0,C_1,.....C_M], |
         *     [U1[C_0,C_1,.....C_M], |
         *              ...           |  x 64
         *     [UN[C_0,C_1,.....C_M]  |
         * ]                          |
         * ===============================>
         * kernel_tm = gemm pack A(kernel_trans)
         *
         * G =
         * [
         *     [1.0f,  0.0f,  0.0f],
         *     [-2/9f, -2/9f, -2/9f],
         *     [-2/9f, 2/9f,  -2/9f],
         *     [1/90f, 1/45f, 2/45f],
         *     [1/90f, -1/45f,2/45f],
         *     [1/45f, 1/90f, 1/180f],
         *     [1/45f, -1/90f,1/180f],
         *     [0.0f,  0.0f,  1.0f]
         * ]
         *
         */
        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f63_transform_and_pack_kernel(const Tensor& kernel, int in_tile_size, Tensor &kernel_tm){

        }

        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f63_transform_and_pack_kernel(const Tensor& kernel, int in_tile_size, Tensor &kernel_tm){
            Shape kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];
            int stride = out_channel * input_channel;

            Tensor kernel_trans(Tensor::InFlow::HOST, kernel_tm.dtype(), kernel_tm.sizes());

            const float *p_kernel = kernel.data<float>();
            int kernel_num_offset = input_channel * 9;
            float *p_kernel_trans = kernel_trans.data<float>();

            const float G[8][3] = {
               {1.0f,  0.0f,  0.0f},
               {-2.f/9, -2.f/9, -2.f/9},
               {-2.f/9, 2.f/9,  -2.f/9},
               {1.f/90, 1.f/45, 2.f/45},
               {1.f/90, -1.f/45,2.f/45},
               {1.f/45, 1.f/90, 1.f/180},
               {1.f/45, -1.f/90,1.f/180},
               {0.0f,  0.0f,  1.0f}
            };

            for (int p = 0; p < out_channel; ++p) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int q = 0; q < input_channel; ++q) {
                    const float *kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    float *kernel_trans_at = p_kernel_trans + p * input_channel + q;

                    // transform kernel
                    const float *k0 = kernel_at;
                    const float *k1 = kernel_at + 3;
                    const float *k2 = kernel_at + 6;

                    float tmp[3][8];
                    for (int i = 0; i < 8; ++i) {
                        tmp[0][i] = k0[0] * G[i][0] + k0[1] * G[i][1] + k0[2] * G[i][2];
                        tmp[1][i] = k1[0] * G[i][0] + k1[1] * G[i][1] + k1[2] * G[i][2];
                        tmp[2][i] = k2[0] * G[i][0] + k2[1] * G[i][1] + k2[2] * G[i][2];
                    }

                    // U,pack:OcIcHW->TOcIc(T for tile count)
                    for (int i = 0; i < 8; ++i) {
                        for (int j = 0; j < 8; ++j) {
                            kernel_trans_at[(i * 8 + j) * stride] = tmp[0][j] * G[i][0] + tmp[1][j] * G[i][1] + tmp[2][j] * G[i][2];
                        }
                    }
                }
            }

            //gemm pack A
            float *kernel_tm_ptr = kernel_tm.data<float>();
            const float* from = p_kernel_trans;
            float *to = kernel_tm_ptr;
            int transform_kernel_tile_offset = out_channel * input_channel;
            for (int i = 0; i < in_tile_size; ++i) {
                math<float,float>::pack8_A(out_channel, input_channel, from, input_channel, to);
                from += transform_kernel_tile_offset;
                to += transform_kernel_tile_offset;
            }
        }

        /*
         * x =
         * [
         *     [Ic_0[0,.....N]],
         *     [Ic_1[0,.....N]],
         *          ...
         *     [Ic_M[0,.....N]],
         * ]
         * ===============================>
         * V = BT(x)B
         * ===============================>
         * x_tm = pack V:NCHW->NTCB(T for tile count,B for tile indices)
         * [                                              |
         *     [Vc_0[tile0_0......tile(tile_count-1)_0]], |
         *     [Vc_1[tile0_0......tile(tile_count-1)_0]], |
         *                     ...                        |  x 64
         *     [Vc_M[tile0_0......tile(tile_count-1)_0]]  |
         * ]                                              |
         *
         * BT =
         * [
         *     [1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f],
         *     [0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f],
         *     [0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f],
         *     [0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f],
         *     [0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f],
         *     [0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f],
         *     [0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f],
         *     [0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f]
         * ]
         *
         */
        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f63_transform_and_pack_input(const Tensor& x, int tile_count, Tensor &x_tm){

        }

        //TODO:maybe simd optimize
        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f63_transform_and_pack_input(const Tensor& x, int tile_count, Tensor &x_tm){
            Shape input_shape = x.sizes();
            int num = input_shape[0];
            int input_channel = input_shape[1];
            int input_height = input_shape[2];
            int input_width = input_shape[3];
            int input_channel_offset = input_height * input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int stride = input_channel * tile_count;
            int out_num_offset = stride * 64;

            const float *input_ptr = x.data<float>();
            float *out_ptr = x_tm.data<float>();

            for (int n = 0; n < num; ++n) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; ++c) {
                    const float *input_cur = input_ptr + n * input_num_offset + c * input_channel_offset;
                    float *out_cur = out_ptr + n * out_num_offset + c * tile_count;

                    int tile_index = 0;
                    float dB[8][8];
                    for (int h = 0; h + 2 < input_height ; h += 6) {
                        for (int w = 0; w + 2 < input_width ; w += 6) {
                            const float *input_at = input_cur + h * input_width + w;

                            for (int i = 0; i < 8; ++i) {
                                const float *i0 = input_at + i * input_width;

                                dB[i][0] = i0[0] - i0[6] + (i0[4] - i0[2]) * 5.25f;
                                dB[i][7] = i0[7] - i0[1] + (i0[3] - i0[5]) * 5.25f;

                                float tmp12_a = (i0[2] + i0[6] - i0[4] * 4.25f);
                                float tmp12_b = (i0[1] + i0[5] - i0[3] * 4.25f);

                                dB[i][1] = tmp12_a + tmp12_b;
                                dB[i][2] = tmp12_a - tmp12_b;

                                float tmp34_a = (i0[6] + i0[2] * 0.25f - i0[4] * 1.25f);
                                float tmp34_b = (i0[1] * 0.5f - i0[3] * 2.5f + i0[5] * 2.f);

                                dB[i][3] = tmp34_a + tmp34_b;
                                dB[i][4] = tmp34_a - tmp34_b;

                                float tmp56_a = (i0[6] + (i0[2] - i0[4] * 1.25f) * 4.f);
                                float tmp56_b = (i0[1] * 2.f - i0[3] * 2.5f + i0[5] * 0.5f);

                                dB[i][5] = tmp56_a + tmp56_b;
                                dB[i][6] = tmp56_a - tmp56_b;
                            }

                            float *out_at = out_cur + tile_index;
                            for (int i = 0; i < 8; ++i) {
                                float dB0, dB1, dB2, dB3, dB4, dB5, dB6, dB7;
                                dB0 = dB[0][i];
                                dB1 = dB[1][i];
                                dB2 = dB[2][i];
                                dB3 = dB[3][i];
                                dB4 = dB[4][i];
                                dB5 = dB[5][i];
                                dB6 = dB[6][i];
                                dB7 = dB[7][i];

                                out_at[i * stride] = dB0 - dB6 + (dB4 - dB2) * 5.25f;
                                out_at[(56 + i) * stride] = dB7 - dB1 + (dB3 - dB5) * 5.25f;

                                float tmp12_a = dB2 + dB6 - dB4 * 4.25f;
                                float tmp12_b = dB1 - dB3 * 4.25f + dB5;

                                out_at[(8 + i) * stride] = tmp12_a + tmp12_b;
                                out_at[(16 + i) * stride] = tmp12_a - tmp12_b;

                                float tmp34_a = dB6 + dB2 * 0.25f - dB4 * 1.25f;
                                float tmp34_b = dB1 * 0.5f - dB3 * 2.5f + dB5 * 2.f;

                                out_at[(24 + i) * stride] = tmp34_a + tmp34_b;
                                out_at[(32 + i) * stride] = tmp34_a - tmp34_b;

                                float tmp56_a = dB6 + (dB2 - dB4 * 1.25f) * 4.f;
                                float tmp56_b = dB1 * 2.f - dB3 * 2.5f + dB5 * 0.5f;

                                out_at[(40 + i) * stride] = tmp56_a + tmp56_b;
                                out_at[(48 + i) * stride] = tmp56_a - tmp56_b;
                            }
                            ++tile_index;
                        }
                    }
                }
            }
        }

        /* out_tm =
         * [                                           |
         *     Out_c_0[tile0_0,tile1_0,....tileN_0],   |
         *     Out_c_1[tile0_0,tile1_0,....tileN_0],   |
         *                  ...                        | x 64
         *     Out_c_M[tile0_0,tile1_0,....tileN_0],   |
         * ]                                           |
         *     N for tile_num
         * =======================================>
         * tmp
         * [
         *     Out_c_0[tile0_0,tile0_1,....tile0_64],
         *     Out_c_0[tile1_0,tile1_1,....tile1_64],
         *                  ...
         *     Out_c_0[tileN_0,tileN_1,....tileN_64],
         *                  ...
         *     Out_c_M[tileN_0,tileN_1,....tileN_64],
         * ]
         * ========================================>
         * out = AT(tmp)A
         * AT =
         * [
         *     [1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f],
         *     [0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f],
         *     [0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f],
         *     [0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f],
         *     [0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f],
         *     [0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f]
         * ]
         */
        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f63_transform_output(const Tensor& out_tm, int tile_count, Tensor& out){

        }

        //TODO:maybe simd optimize
        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f63_transform_output(const Tensor& out_tm, int tile_count, Tensor& out){
            Shape out_shape = out.sizes();
            Shape out_tm_shape = out_tm.sizes();
            int num = out_tm_shape[0];
            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];
            int stride = out_channel * tile_count;
            int out_tm_num_offset = 64 * stride;
            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const float* out_tm_ptr = out_tm.data<float>();
            float *out_ptr = out.data<float>();

            for (int n = 0; n < num; ++n) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < out_channel; ++c) {
                    float tmpA[8][6];
                    int tile_offset = 0;
                    const float *out_tm_cur = out_tm_ptr + n * out_tm_num_offset + c * tile_count;
                    float *out_cur = out_ptr + n * out_num_offset + c * out_channel_offset;

                    for (int h = 0; h + 5 < out_height; h += 6) {
                        for (int w = 0; w + 5 < out_width; w += 6) {
                            const float *out_tm_at = out_tm_cur + tile_offset;
                            float *out_at = out_cur + h * out_width + w;

                            for (int i = 0; i < 8; ++i) {
                                float t0,t1,t2,t3,t4,t5,t6,t7;
                                t0 = out_tm_at[0];
                                t1 = out_tm_at[1 * stride];
                                t2 = out_tm_at[2 * stride];
                                t3 = out_tm_at[3 * stride];
                                t4 = out_tm_at[4 * stride];
                                t5 = out_tm_at[5 * stride];
                                t6 = out_tm_at[6 * stride];
                                t7 = out_tm_at[7 * stride];

                                float re0 = t1 + t2;
                                float re1 = t1 - t2;
                                float re2 = t3 + t4;
                                float re3 = t3 - t4;
                                float re4 = t5 + t6;
                                float re5 = t5 - t6;

                                tmpA[i][0] = t0 + re0 + re2 + re4 * 32.f;
                                tmpA[i][1] = re1 + re3 + re3 + re5 * 16.f;
                                tmpA[i][2] = re0 + re2 * 4 + re4 * 8.f;
                                tmpA[i][3] = re1 + re3 * 8 + re5 * 4.f;
                                tmpA[i][4] = re0 + re2 * 16 + re4 + re4;
                                tmpA[i][5] = re1 + re3 * 32 + re5 + t7;

                                out_tm_at += 8 * stride;
                            }

                            float *out0 = out_at;
                            float *out1 = out0 + out_width;
                            float *out2 = out1 + out_width;
                            float *out3 = out2 + out_width;
                            float *out4 = out3 + out_width;
                            float *out5 = out4 + out_width;

                            for (int i = 0; i < 6; ++i) {
                                float t0,t1,t2,t3,t4,t5,t6,t7;
                                t0 = tmpA[0][i];
                                t1 = tmpA[1][i];
                                t2 = tmpA[2][i];
                                t3 = tmpA[3][i];
                                t4 = tmpA[4][i];
                                t5 = tmpA[5][i];
                                t6 = tmpA[6][i];
                                t7 = tmpA[7][i];

                                float re0 = t1 + t2;
                                float re1 = t1 - t2;
                                float re2 = t3 + t4;
                                float re3 = t3 - t4;
                                float re4 = t5 + t6;
                                float re5 = t5 - t6;

                                out0[i] = t0 + re0 + re2 + re4 * 32.f;
                                out1[i] = re1 + re3 + re3 + re5 * 16.f;
                                out2[i] = re0 + re2 * 4 + re4 * 8.f;
                                out3[i] = re1 + re3 * 8 + re5 * 4.f;
                                out4[i] = re0 + re2 * 16 + re4 + re4;
                                out5[i] = re1 + re3 * 32 + re5 + t7;
                            }
                            ++tile_offset;
                        }
                    }
                }
            }
        }

        template <typename T>
        void Conv2dWinogradAlgorithm<T>::winograd_f63(const Tensor &x,
                                             const Padding2D &padding,
                                             float padding_value,
                                             const Tensor &kernel,
                                             Tensor &out,
                                             bool kernel_transformed){

        }

        template <>
        void Conv2dWinogradAlgorithm<float>::winograd_f63(const Tensor &x,
                                             const Padding2D &padding,
                                             float padding_value,
                                             const Tensor &kernel,
                                             Tensor &out,
                                             bool kernel_transformed){
            int tile_width = 6;
            int tile_height = 6;

            Shape kernel_shape = kernel.sizes();
            Tensor input_padded = x;
            Tensor out_padded = out;
            bool out_padded_flag = false;
            Stride2D stride(1, 1);
            KSize2D ksize(3, 3);
            KernelCommonFunc<float>::in_out_pad_and_fix_size(x,
                                                             kernel_shape,
                                                             out,
                                                             tile_height,
                                                             tile_width,
                                                             padding,
                                                             padding_value,
                                                             stride,
                                                             ksize,
                                                             input_padded,
                                                             out_padded,
                                                             out_padded_flag);

            auto output_shape = out_padded.sizes();
            auto input_shape = input_padded.sizes();
            auto src_output_shape = out.sizes();

            int num = input_shape[0];
            int input_channel = input_shape[1];
            int out_channel = output_shape[1];
            int out_height = output_shape[2];
            int out_width = output_shape[3];

            int tile_block_height = out_height / tile_height;
            int tile_block_width = out_width / tile_width;
            int tile_block_num = tile_block_height * tile_block_width;
            int in_tile_size = (tile_width + 2) * (tile_height + 2);

            //transform kernel
            Tensor kernel_tm = kernel;
            if(!kernel_transformed){
                Shape kernel_tm_shape = {in_tile_size, kernel_shape[0], kernel_shape[1]};
                Tensor kernel_tmp(Tensor::InFlow::HOST, kernel.dtype(), kernel_tm_shape);
//                std::memset(kernel_tmp.data<float>(), 0, sizeof(float)*kernel_tmp.count());
                winograd_f63_transform_and_pack_kernel(kernel, in_tile_size, kernel_tmp);
                kernel_tm = kernel_tmp;
            }
            const float *kernel_ptr = kernel_tm.data<float>();

            //transform input
            Shape input_tm_shape = {num, in_tile_size, input_channel, tile_block_num};
            Tensor input_tm(Tensor::InFlow::HOST, input_padded.dtype(), input_tm_shape);
            winograd_f63_transform_and_pack_input(input_padded, tile_block_num, input_tm);
            const float *trans_input_ptr = input_tm.data<float>();

            //eltwise_gemm->gemm(O = U*V)
            Shape transform_out_shape = {num, in_tile_size, out_channel, tile_block_num};
            Tensor transform_out(Tensor::InFlow::HOST, out.dtype(), transform_out_shape);
            float *trans_out_ptr = transform_out.data<float>();

            int transform_kernel_tile_offset = out_channel * input_channel;
            int transform_input_tile_offset = input_channel * tile_block_num;
            int transform_input_num_offset = in_tile_size * transform_input_tile_offset;
            int transform_out_tile_offset = out_channel * tile_block_num;
            int transform_out_num_offset = in_tile_size * transform_out_tile_offset;

            for (int n = 0; n < num; ++n) {
                const float *trans_input_cur = trans_input_ptr + n * transform_input_num_offset;
                float *trans_out_cur = trans_out_ptr + n * transform_out_num_offset;

                for (int i = 0; i < in_tile_size; ++i) {
                    const float *trans_kernel_at = kernel_ptr + i * transform_kernel_tile_offset;
                    const float *trans_input_at = trans_input_cur + i * transform_input_tile_offset;
                    float *trans_out_at = trans_out_cur + i * transform_out_tile_offset;
                    math<float, float>::gemm(out_channel,
                                             tile_block_num,
                                             input_channel,
                                             1.f,
                                             trans_kernel_at,
                                             trans_input_at,
                                             0.f,
                                             trans_out_at,
                                             false,
                                             true);
                }
            }

            //transform output
            winograd_f63_transform_output(transform_out, tile_block_num, out_padded);

            //cut output
            if (out_padded_flag) {
                std::array<int, 2> pad_h = {0, src_output_shape[2] - out_height};
                std::array<int, 2> pad_w = {0, src_output_shape[3] - out_width};
                PadAlgorithm<float>::cut2d(out_padded, pad_h, pad_w, 0.f, out);
            }
        }
    } //cpu
} //ts

