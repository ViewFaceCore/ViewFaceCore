#include "kernels/cpu/conv2d_algorithm.h"

#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif

#include "kernels/common/simd.h"

namespace ts {
    namespace cpu {

        template<typename T>
        static void inner_transpose_temp(size_t m, size_t n, T *in, T *out) //  A[m][n] -> A[n][m]
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    out[j * m + i] = in[i * n + j];
        }

        template<typename T>
        static void inner_cut(const Tensor& src, Tensor& dst, int cut_top, int cut_bottom, int cut_left, int cut_right) {

            auto src_shape = src.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = dst.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h - cut_top - cut_bottom;
            int out_w = src_w - cut_left - cut_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            dst.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = src.data<T>();
            T* dst_data = dst.data<T>();

            for (int n = 0; n < num; n++) {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < channel; c++) {
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    const T* cut_at = src_at + cut_top * src_w + cut_left;

                    for (int h = 0; h < out_h; h++) {
                        if (out_w <12) {
                            for (int w = 0; w < out_w; w++)
                            {
                                dst_at[w] = cut_at[w];
                            }
                        }
                        else {
                            std::memcpy(dst_at, cut_at, out_w * sizeof(T));
                        }

                        dst_at += out_w;
                        cut_at += src_w;
                    }
                }
            }
        }

        template<typename T>
        static void inner_pad(const Tensor& src, Tensor& dst, int pad_top, int pad_bottom, int pad_left, int pad_right, T pad_value) {

            auto src_shape = src.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = dst.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h + pad_top + pad_bottom;
            int out_w = src_w + pad_left + pad_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            dst.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = src.data<T>();
            T* dst_data = dst.data<T>();

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < channel; c++)
                {
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    //fill top
                    int y = 0;
                    for (; y < pad_top; y++)
                    {
                        int x = 0;
                        for (; x < out_w; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        dst_at += out_w;
                    }

                    //fill center
                    for (; y < pad_top + src_shape[2]; y++)
                    {
                        int x = 0;
                        for (; x < pad_left; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        if (src_w < 12)
                        {
                            for (; x < (pad_left + src_w); x++)
                            {
                                dst_at[x] = src_at[x - pad_left];
                            }
                        }
                        else
                        {
                            std::memcpy(dst_at + pad_left, src_at, src_w * sizeof(T));
                            x += src_w;
                        }
                        for (; x < out_w; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        dst_at += out_w;
                        src_at += src_w;
                    }
                    // fill bottom
                    for (; y < out_h; y++)
                    {
                        int x = 0;
                        for (; x < out_w; x++)
                        {
                            dst_at[x] = pad_value;
                        }
                        dst_at += out_w;
                    }
                }
            }
        }

        static inline void winograd_f63_input_transform(
            float32x4 &r0,
            float32x4 &r1,
            float32x4 &r2,
            float32x4 &r3,
            float32x4 &r4,
            float32x4 &r5,
            float32x4 &r6,
            float32x4 &r7,
            float32x4 &t1,
            float32x4 &t2,
            float32x4 &m1,
            float32x4 &m2,
            float32x4 &p1,
            float32x4 &p2,
            const float32x4 &f5_25,
            const float32x4 &f4_25,
            const float32x4 &f4,
            const float32x4 &f2_5,
            const float32x4 &f2,
            const float32x4 &f1_25,
            const float32x4 &f0_5,
            const float32x4 &f0_25
        )
        {
            r0 = r0 - r6 + (r4 - r2) * f5_25;
            r7 = r7 - r1 + (r3 - r5) * f5_25;

            t1 = r2 + r6 - r4 * f4_25;
            t2 = r1 + r5 - r3 * f4_25;

            m1 = r4 * f1_25;
            m2 = r3 * f2_5;

            p1 = r6 + (r2 * f0_25 - m1);
            p2 = r1 * f0_5 - m2 + r5 * f2;

            r3 = p1 + p2;
            r4 = p1 - p2;

            p1 = r6 + (r2 - m1) * f4;
            p2 = r1 * f2 - m2 + r5 * f0_5;

            r5 = p1 + p2;
            r6 = p1 - p2;

            r1 = t1 + t2;
            r2 = t1 - t2;
        }

        static inline void winograd_f63_output_transform(
            float32x4 &m0,
            float32x4 &m1,
            float32x4 &m2,
            float32x4 &m3,
            float32x4 &m4,
            float32x4 &m5,
            float32x4 &m6,
            float32x4 &m7,
            float32x4 &m1_add_m2,
            float32x4 &m1_sub_m2,
            float32x4 &m3_add_m4,
            float32x4 &m3_sub_m4,
            float32x4 &m5_add_m6,
            float32x4 &m5_sub_m6,
            const float32x4& f0,
            const float32x4& f2,
            const float32x4& f4,
            const float32x4& f8,
            const float32x4& f16,
            const float32x4& f32
        )
        {
            /*
            * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
            * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
            * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
            * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
            * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
            * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
            */

            m1_add_m2 = m1 + m2;
            m1_sub_m2 = m1 - m2;
            m3_add_m4 = m3 + m4;
            m3_sub_m4 = m3 - m4;
            m5_add_m6 = m5 + m6;
            m5_sub_m6 = m5 - m6;

            m0 = m0 + m1_add_m2;
            m5 = m7 + m1_sub_m2;

            m1 = fmadd(f16, m5_sub_m6, m1_sub_m2);
            m4 = fmadd(f16, m3_add_m4, m1_add_m2);

            m2 = fmadd(f8, m5_add_m6, m1_add_m2);
            m3 = fmadd(f8, m3_sub_m4, m1_sub_m2);

            m0 = fmadd(f32, m5_add_m6, m0);
            m0 = m0 + m3_add_m4;

            m5 = fmadd(f32, m3_sub_m4, m5);
            m5 = m5 + m5_sub_m6;

            m1 = fmadd(m3_sub_m4, f2, m1);
            m4 = fmadd(m5_add_m6, f2, m4);

            m2 = fmadd(m3_add_m4, f4, m2);
            m3 = fmadd(m5_sub_m6, f4, m3);

            m6 = f0;
            m7 = f0;
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd23_transform_kernel(const Tensor& kernel, Tensor &kernel_tm) {
            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 16;

            const T ktm[4][3] = {
                { 1,     0,     0 },
                { T(1) / 2,   T(1) / 2,   T(1) / 2 },
                { T(1) / 2,   -T(1) / 2,   T(1) / 2 },
                { 0,     0,     1 }
            };

            for (int p = 0; p < out_channel; p++)
            {
                for (int q = 0; q < input_channel; q++)
                {
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 16;

                    // transform kernel
                    const T* k0 = kernel_at;
                    const T* k1 = kernel_at + 3;
                    const T* k2 = kernel_at + 6;

                    T tmp[4][3];
                    for (int i = 0; i<4; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // UT
                    for (int j = 0; j<4; j++)
                    {
                        T* tmpp = &tmp[j][0];

                        for (int i = 0; i<4; i++)
                        {
                            kernel_tm_at[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd23_transform_kernel_inplace(const Tensor& kernel, Tensor &kernel_tm) {

            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 16;

            const T ktm[12] = {
                1,     0,     0,
                T(1) / 2,   T(1) / 2,   T(1) / 2,
                T(1) / 2,   -T(1) / 2,   T(1) / 2,
                0,     0,     1
            };

            for (int p = 0; p < out_channel; p++) {
                for (int q = 0; q < input_channel; q++) {
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 16;

                    T tmp_mid[12], tmp_out[12];
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 3, 3, T(1), ktm, kernel_at, 0, tmp_mid);
                    #else
                    cpu::math<T,T>::gemm(blas::NoTrans, blas::NoTrans, 4, 3, 3, T(1), ktm, kernel_at, 0, tmp_mid);
                    #endif
                    inner_transpose_temp<T>(4, 3, tmp_mid, tmp_out);
                    //inner_transpose_temp<T>(4, 3, (T*)ktm, tmp_out);
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 4, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    //cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 4, 3, T(1), (const T*)tmp_out,ktm, 0, kernel_tm_at);
                    #else
                    cpu::math<T,T>::gemm(blas::NoTrans, blas::NoTrans, 4, 3, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    #endif
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd63_transform_kernel(const Tensor& kernel, Tensor &kernel_tm) {
            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 64;

            const T ktm[8][3] = {
                { T(1),     0,     0 },
                { -T(2) / 9,  -T(2) / 9,  -T(2) / 9 },
                { -T(2) / 9,   T(2) / 9,  -T(2) / 9 },
                { T(1) / 90,  T(1) / 45,  T(2) / 45 },
                { T(1) / 90, -T(1) / 45,  T(2) / 45 },
                { T(1) / 45,  T(1) / 90, T(1) / 180 },
                { T(1) / 45, -T(1) / 90, T(1) / 180 },
                { 0,     0,     1 }
            };

            for (int p = 0; p < out_channel; p++)
            {
                for (int q = 0; q < input_channel; q++)
                {
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 64;

                    // transform kernel
                    const T* k0 = kernel_at;
                    const T* k1 = kernel_at + 3;
                    const T* k2 = kernel_at + 6;

                    T tmp[8][3];
                    for (int i = 0; i<8; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // U
                    for (int j = 0; j<8; j++)
                    {
                        T* tmpp = &tmp[j][0];

                        for (int i = 0; i<8; i++)
                        {
                            kernel_tm_at[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd63_transform_kernel_inplace(const Tensor& kernel, Tensor &kernel_tm) {

            auto kernel_shape = kernel.sizes();
            int out_channel = kernel_shape[0];
            int input_channel = kernel_shape[1];

            const T* p_kernel = kernel.data<T>();
            int kernel_num_offset = input_channel * 9;
            T* p_kernel_tm = kernel_tm.data<T>();
            int kernel_tm_num_offset = input_channel * 64;

            const T ktm[24] =
            {
                1, 0, 0,
                -T(2) / 9, -T(2) / 9, -T(2) / 9,
                -T(2) / 9, T(2) / 9, -T(2) / 9,
                T(1) / 90, T(1) / 45, T(2) / 45,
                T(1) / 90, -T(1) / 45, T(2) / 45,
                T(1) / 45, T(1) / 90, T(1) / 180,
                T(1) / 45, -T(1) / 90, T(1) / 180,
                0, 0, 1
            };

            for (int p = 0; p < out_channel; p++) {
                for (int q = 0; q < input_channel; q++) {
                    const T* kernel_at = p_kernel + p * kernel_num_offset + q * 9;
                    T* kernel_tm_at = p_kernel_tm + p * kernel_tm_num_offset + q * 64;

                    T tmp_mid[24], tmp_out[24];
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 8, 3, 3, T(1), ktm, kernel_at, 0, tmp_mid);
                    #else
                    cpu::math<T, T>::gemm(blas::NoTrans, blas::NoTrans, 8, 3, 3, T(1), ktm, kernel_at, 0, tmp_mid);
                    #endif
                    inner_transpose_temp<T>(8, 3, tmp_mid, tmp_out);
                    //inner_transpose_temp<T>(4, 3, (T*)ktm, tmp_out);
                    #ifdef TS_USE_CBLAS
                    cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 8, 8, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    //cblas::math<T>::gemm(blas::NoTrans, blas::NoTrans, 4, 4, 3, T(1), (const T*)tmp_out,ktm, 0, kernel_tm_at);
                    #else
                    cpu::math<T, T>::gemm(blas::NoTrans, blas::NoTrans, 8, 8, 3, T(1), ktm, (const T*)tmp_out, 0, kernel_tm_at);
                    #endif
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd23(const Tensor &x, const Tensor &k_tm, Tensor &out) {

            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 1) / 2 * 2;
            output_h = (output_h + 1) / 2 * 2;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<T>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            // const float BT[4][4] = {
            //     {1.0f,  0.0f, -1.0f,  0.0f},
            //     {0.0f,  1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  0.00f, 1.0f}
            // };   

            int w_tm = output_w / 2 * 4;
            int h_tm = output_h / 2 * 4;
            int col_blocks = w_tm / 4;
            int row_blocks = h_tm / 4;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 16 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 16 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const T* src_ptr = input_bordered.data<T>();
            T* dst_ptr = input_tm.data<T>();
            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; c++)
                {
                    const T* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    T* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        const T* r0 = src_at + i * input_padded_w * 2;
                        const T* r1 = r0 + input_padded_w;
                        const T* r2 = r1 + input_padded_w;
                        const T* r3 = r2 + input_padded_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            T d0[4], d1[4], d2[4], d3[4];
                            T w0[4], w1[4], w2[4], w3[4];
                            T t0[4], t1[4], t2[4], t3[4];

                            for (int n = 0; n < 4; n++)
                            {
                                d0[n] = r0[n];
                                d1[n] = r1[n];
                                d2[n] = r2[n];
                                d3[n] = r3[n];
                            }

                            // BT * d * B == (BT * (BT*d)T)T

                            // w = BT * d
                            for (int n = 0; n < 4; n++)
                            {
                                w0[n] = d0[n] - d2[n];
                                w1[n] = d1[n] + d2[n];
                                w2[n] = d2[n] - d1[n];
                                w3[n] = d3[n] - d1[n];
                            }

                            // w to wT
                            {
                                t0[0] = w0[0]; t1[0] = w0[1]; t2[0] = w0[2]; t3[0] = w0[3];
                                t0[1] = w1[0]; t1[1] = w1[1]; t2[1] = w1[2]; t3[1] = w1[3];
                                t0[2] = w2[0]; t1[2] = w2[1]; t2[2] = w2[2]; t3[2] = w2[3];
                                t0[3] = w3[0]; t1[3] = w3[1]; t2[3] = w3[2]; t3[3] = w3[3];
                            }

                            // d = BT * wT 
                            for (int n = 0; n < 4; n++)
                            {
                                d0[n] = t0[n] - t2[n];
                                d1[n] = t1[n] + t2[n];
                                d2[n] = t2[n] - t1[n];
                                d3[n] = t3[n] - t1[n];
                            }

                            for (int n = 0; n < 4; n++)
                            {
                                dst_at[n] = d0[n];
                                dst_at[n + 4] = d1[n];
                                dst_at[n + 8] = d2[n];
                                dst_at[n + 12] = d3[n];
                            }

                            r0 += 2;
                            r1 += 2;
                            r2 += 2;
                            r3 += 2;

                            dst_at += 16;

                        }
                    }
                }

            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 16 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 16;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            T* out_tm_ptr = output_tm.data<T>();
            T* input_tm_ptr = dst_ptr;

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_tm_1 = out_tm_0 + outtm_c_offset;
                    T* out_tm_2 = out_tm_1 + outtm_c_offset;
                    T* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();

                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const T* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const T* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const T* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 16;
                        T* out_1 = out_tm_1 + i * 16;
                        T* out_2 = out_tm_2 + i * 16;
                        T* out_3 = out_tm_3 + i * 16;

                        T sum_0[16] = { T(0) };
                        T sum_1[16] = { T(0) };
                        T sum_2[16] = { T(0) };
                        T sum_3[16] = { T(0) };

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;

                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r1[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r2[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r3[k] * k0[k];
                                k0 -= 48;

                                sum_1[k] += r0[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r1[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r2[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r3[k] * k1[k];
                                k1 -= 48;

                                sum_2[k] += r0[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r1[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r2[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r3[k] * k2[k];
                                k2 -= 48;

                                sum_3[k] += r0[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r1[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r2[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r3[k] * k3[k];
                                k3 -= 48;
                            }
                        }

                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_1[k] += r0[k] * k1[k];
                                sum_2[k] += r0[k] * k2[k];
                                sum_3[k] += r0[k] * k3[k];
                            }
                        }

                        for (int k = 0; k < 16; k++)
                        {
                            out_0[k] = sum_0[k];
                            out_1[k] = sum_1[k];
                            out_2[k] = sum_2[k];
                            out_3[k] = sum_3[k];
                        }

                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = remain_outch; c < output_channel; c++)
                {
                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();
                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 16;
                        T sum_0[16] = { T(0) };

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = k0 + ktm_c_offset;
                            const T* k2 = k1 + ktm_c_offset;
                            const T* k3 = k2 + ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_0[k] += r1[k] * k1[k];
                                sum_0[k] += r2[k] * k2[k];
                                sum_0[k] += r3[k] * k3[k];
                            }
                        }

                        for (; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                            }
                        }

                        for (int k = 0; k < 16; k++)
                        {
                            out_0[k] = sum_0[k];
                        }

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            T* out_ptr = output_bordered.data<T>();

            // const float AT[2][4] = {
            //     {1.0f,  1.0f,  1.0f,  0.0f},
            //     {0.0f,  1.0f, -1.0f,  1.0f}
            // }; 

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < output_channel; c++)
                {
                    T* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        T* out_0 = out_at + i * output_w * 2;
                        T* out_1 = out_0 + output_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            T* out_tile = output_tm_at + (i * col_blocks + j) * 16;

                            T s0[4], s1[4], s2[4], s3[4];
                            T w0[4], w1[4];
                            T d0[2], d1[2], d2[2], d3[2];
                            T o0[2], o1[2];

                            for (int n = 0; n < 4; n++)
                            {
                                s0[n] = out_tile[n];
                                s1[n] = out_tile[n + 4];
                                s2[n] = out_tile[n + 8];
                                s3[n] = out_tile[n + 12];
                            }
                            // w = A_T * W
                            for (int n = 0; n < 4; n++)
                            {
                                w0[n] = s0[n] + s1[n] + s2[n];
                                w1[n] = s1[n] - s2[n] + s3[n];
                            }
                            // transpose w to w_t
                            {
                                d0[0] = w0[0]; d0[1] = w1[0];
                                d1[0] = w0[1]; d1[1] = w1[1];
                                d2[0] = w0[2]; d2[1] = w1[2];
                                d3[0] = w0[3]; d3[1] = w1[3];
                            }
                            // Y = A_T * w_t
                            for (int n = 0; n < 2; n++)
                            {
                                o0[n] = d0[n] + d1[n] + d2[n];
                                o1[n] = d1[n] - d2[n] + d3[n];
                            }

                            out_0[0] = o0[0];
                            out_0[1] = o0[1];
                            out_1[0] = o1[0];
                            out_1[1] = o1[1];

                            out_0 += 2;
                            out_1 += 2;
                        }
                    }
                }
            }

            inner_cut<T>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }

#ifdef TS_USE_SIMD
        template<>
        void Conv2dAlgorithm<float>::conv3x3_winograd23(const Tensor &x, const Tensor &k_tm, Tensor &out) {

            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 1) / 2 * 2;
            output_h = (output_h + 1) / 2 * 2;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<float>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            // const float BT[4][4] = {
            //     {1.0f,  0.0f, -1.0f,  0.0f},
            //     {0.0f,  1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  0.00f, 1.0f}
            // };   

            int w_tm = output_w / 2 * 4;
            int h_tm = output_h / 2 * 4;
            int col_blocks = w_tm / 4;
            int row_blocks = h_tm / 4;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 16 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 16 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const float* src_ptr = input_bordered.data<float>();
            float* dst_ptr = input_tm.data<float>();
            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; c++)
                {
                    const float* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    float* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        const float* r0 = src_at + i * input_padded_w * 2;
                        const float* r1 = r0 + input_padded_w;
                        const float* r2 = r1 + input_padded_w;
                        const float* r3 = r2 + input_padded_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            float32x4 dst0(dst_at);
                            float32x4 dst1(dst_at + 4);
                            float32x4 dst2(dst_at + 8);
                            float32x4 dst3(dst_at + 12);

                            float32x4 d0(r0), d1(r1), d2(r2), d3(r3);

                            // BT * d * B == (BT * (BT*d)T)T
                            // w = BT * d
                            float32x4 w0 = d0 - d2;
                            float32x4 w1 = d1 + d2;
                            float32x4 w2 = d2 - d1;
                            float32x4 w3 = d3 - d1;

                            // w to wT
                            //_MM_TRANSPOSE4_PS(w0.value,w1.value,w2.value,w3.value);
                            transposex4x4(w0, w1, w2, w3);
                            // d = BT * wT
                            d0 = w0 - w2;
                            d1 = w1 + w2;
                            d2 = w2 - w1;
                            d3 = w3 - w1;

                            d0.store(dst_at);
                            d1.store(dst_at + 4);
                            d2.store(dst_at + 8);
                            d3.store(dst_at + 12);

                            r0 += 2;
                            r1 += 2;
                            r2 += 2;
                            r3 += 2;

                            dst_at += 16;

                        }
                    }
                }

            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 16 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 16;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            float* out_tm_ptr = output_tm.data<float>();
            float* input_tm_ptr = dst_ptr;

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    float* out_tm_1 = out_tm_0 + outtm_c_offset;
                    float* out_tm_2 = out_tm_1 + outtm_c_offset;
                    float* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const float* kernel_tm_ptr = k_tm.data<float>();

                    const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const float* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const float* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const float* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        float* out_0 = out_tm_0 + i * 16;
                        float* out_1 = out_tm_1 + i * 16;
                        float* out_2 = out_tm_2 + i * 16;
                        float* out_3 = out_tm_3 + i * 16;

                        float32x4 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f);
                        float32x4 sum10(0.f), sum11(0.f), sum12(0.f), sum13(0.f);
                        float32x4 sum20(0.f), sum21(0.f), sum22(0.f), sum23(0.f);
                        float32x4 sum30(0.f), sum31(0.f), sum32(0.f), sum33(0.f);

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;

                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 16;
                            const float* r1 = r0 + tm_c_offset;
                            const float* r2 = r1 + tm_c_offset;
                            const float* r3 = r2 + tm_c_offset;
                            float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);
                            float32x4 r10(r1), r11(r1 + 4), r12(r1 + 8), r13(r1 + 12);
                            float32x4 r20(r2), r21(r2 + 4), r22(r2 + 8), r23(r2 + 12);
                            float32x4 r30(r3), r31(r3 + 4), r32(r3 + 8), r33(r3 + 12);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const float* k3 = kernel_tm_3 + q * ktm_c_offset;
                            float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);
                            float32x4 k010(k0 + 16), k011(k0 + 20), k012(k0 + 24), k013(k0 + 28);
                            float32x4 k020(k0 + 32), k021(k0 + 36), k022(k0 + 40), k023(k0 + 44);
                            float32x4 k030(k0 + 48), k031(k0 + 52), k032(k0 + 56), k033(k0 + 60);

                            float32x4 k100(k1), k101(k1 + 4), k102(k1 + 8), k103(k1 + 12);
                            float32x4 k110(k1 + 16), k111(k1 + 20), k112(k1 + 24), k113(k1 + 28);
                            float32x4 k120(k1 + 32), k121(k1 + 36), k122(k1 + 40), k123(k1 + 44);
                            float32x4 k130(k1 + 48), k131(k1 + 52), k132(k1 + 56), k133(k1 + 60);

                            float32x4 k200(k2), k201(k2 + 4), k202(k2 + 8), k203(k2 + 12);
                            float32x4 k210(k2 + 16), k211(k2 + 20), k212(k2 + 24), k213(k2 + 28);
                            float32x4 k220(k2 + 32), k221(k2 + 36), k222(k2 + 40), k223(k2 + 44);
                            float32x4 k230(k2 + 48), k231(k2 + 52), k232(k2 + 56), k233(k2 + 60);

                            float32x4 k300(k3), k301(k3 + 4), k302(k3 + 8), k303(k3 + 12);
                            float32x4 k310(k3 + 16), k311(k3 + 20), k312(k3 + 24), k313(k3 + 28);
                            float32x4 k320(k3 + 32), k321(k3 + 36), k322(k3 + 40), k323(k3 + 44);
                            float32x4 k330(k3 + 48), k331(k3 + 52), k332(k3 + 56), k333(k3 + 60);

                            //sum00 += r00 * k000; sum00 += r10 * k010; sum00 += r20 * k020; sum00 += r30 * k030;
                            //sum01 += r01 * k001; sum01 += r11 * k011; sum01 += r21 * k021; sum01 += r31 * k031;
                            //sum02 += r02 * k002; sum02 += r12 * k012; sum02 += r22 * k022; sum02 += r32 * k032;
                            //sum03 += r03 * k003; sum03 += r13 * k013; sum03 += r23 * k023; sum03 += r33 * k033;

                            //sum10 += r00 * k100; sum10 += r10 * k110; sum10 += r20 * k120; sum10 += r30 * k130;
                            //sum11 += r01 * k101; sum11 += r11 * k111; sum11 += r21 * k121; sum11 += r31 * k131;
                            //sum12 += r02 * k102; sum12 += r12 * k112; sum12 += r22 * k122; sum12 += r32 * k132;
                            //sum13 += r03 * k103; sum13 += r13 * k113; sum13 += r23 * k123; sum13 += r33 * k133;
                            //
                            //sum20 += r00 * k200; sum20 += r10 * k210; sum20 += r20 * k220; sum20 += r30 * k230;
                            //sum21 += r01 * k201; sum21 += r11 * k211; sum21 += r21 * k221; sum21 += r31 * k231;
                            //sum22 += r02 * k202; sum22 += r12 * k212; sum22 += r22 * k222; sum22 += r32 * k232;
                            //sum23 += r03 * k203; sum23 += r13 * k213; sum23 += r23 * k223; sum23 += r33 * k233;

                            //sum30 += r00 * k300; sum30 += r10 * k310; sum30 += r20 * k320; sum30 += r30 * k330;
                            //sum31 += r01 * k301; sum31 += r11 * k311; sum31 += r21 * k321; sum31 += r31 * k331;
                            //sum32 += r02 * k302; sum32 += r12 * k312; sum32 += r22 * k322; sum32 += r32 * k332;
                            //sum33 += r03 * k303; sum33 += r13 * k313; sum33 += r23 * k323; sum33 += r33 * k333;

                            sum00 = fmadd(r00, k000, sum00); sum00 = fmadd(r10, k010, sum00); sum00 = fmadd(r20, k020, sum00); sum00 = fmadd(r30, k030, sum00);
                            sum01 = fmadd(r01, k001, sum01); sum01 = fmadd(r11, k011, sum01); sum01 = fmadd(r21, k021, sum01); sum01 = fmadd(r31, k031, sum01);
                            sum02 = fmadd(r02, k002, sum02); sum02 = fmadd(r12, k012, sum02); sum02 = fmadd(r22, k022, sum02); sum02 = fmadd(r32, k032, sum02);
                            sum03 = fmadd(r03, k003, sum03); sum03 = fmadd(r13, k013, sum03); sum03 = fmadd(r23, k023, sum03); sum03 = fmadd(r33, k033, sum03);

                            sum10 = fmadd(r00, k100, sum10); sum10 = fmadd(r10, k110, sum10); sum10 = fmadd(r20, k120, sum10); sum10 = fmadd(r30, k130, sum10);
                            sum11 = fmadd(r01, k101, sum11); sum11 = fmadd(r11, k111, sum11); sum11 = fmadd(r21, k121, sum11); sum11 = fmadd(r31, k131, sum11);
                            sum12 = fmadd(r02, k102, sum12); sum12 = fmadd(r12, k112, sum12); sum12 = fmadd(r22, k122, sum12); sum12 = fmadd(r32, k132, sum12);
                            sum13 = fmadd(r03, k103, sum13); sum13 = fmadd(r13, k113, sum13); sum13 = fmadd(r23, k123, sum13); sum13 = fmadd(r33, k133, sum13);

                            sum20 = fmadd(r00, k200, sum20); sum20 = fmadd(r10, k210, sum20); sum20 = fmadd(r20, k220, sum20); sum20 = fmadd(r30, k230, sum20);
                            sum21 = fmadd(r01, k201, sum21); sum21 = fmadd(r11, k211, sum21); sum21 = fmadd(r21, k221, sum21); sum21 = fmadd(r31, k231, sum21);
                            sum22 = fmadd(r02, k202, sum22); sum22 = fmadd(r12, k212, sum22); sum22 = fmadd(r22, k222, sum22); sum22 = fmadd(r32, k232, sum22);
                            sum23 = fmadd(r03, k203, sum23); sum23 = fmadd(r13, k213, sum23); sum23 = fmadd(r23, k223, sum23); sum23 = fmadd(r33, k233, sum23);

                            sum30 = fmadd(r00, k300, sum30); sum30 = fmadd(r10, k310, sum30); sum30 = fmadd(r20, k320, sum30); sum30 = fmadd(r30, k330, sum30);
                            sum31 = fmadd(r01, k301, sum31); sum31 = fmadd(r11, k311, sum31); sum31 = fmadd(r21, k321, sum31); sum31 = fmadd(r31, k331, sum31);
                            sum32 = fmadd(r02, k302, sum32); sum32 = fmadd(r12, k312, sum32); sum32 = fmadd(r22, k322, sum32); sum32 = fmadd(r32, k332, sum32);
                            sum33 = fmadd(r03, k303, sum33); sum33 = fmadd(r13, k313, sum33); sum33 = fmadd(r23, k323, sum33); sum33 = fmadd(r33, k333, sum33);

                        }

                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 16;
                            float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const float* k3 = kernel_tm_3 + q * ktm_c_offset;

                            float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);

                            float32x4 k100(k1), k101(k1 + 4), k102(k1 + 8), k103(k1 + 12);

                            float32x4 k200(k2), k201(k2 + 4), k202(k2 + 8), k203(k2 + 12);

                            float32x4 k300(k3), k301(k3 + 4), k302(k3 + 8), k303(k3 + 12);

                            //sum00 += r00 * k000;
                            //sum01 += r01 * k001;
                            //sum02 += r02 * k002;
                            //sum03 += r03 * k003;

                            //sum10 += r00 * k100;
                            //sum11 += r01 * k101;
                            //sum12 += r02 * k102;
                            //sum13 += r03 * k103;

                            //sum20 += r00 * k200;
                            //sum21 += r01 * k201;
                            //sum22 += r02 * k202;
                            //sum23 += r03 * k203;

                            //sum30 += r00 * k300;
                            //sum31 += r01 * k301;
                            //sum32 += r02 * k302;
                            //sum33 += r03 * k303;

                            sum00 = fmadd(r00, k000, sum00);
                            sum01 = fmadd(r01, k001, sum01);
                            sum02 = fmadd(r02, k002, sum02);
                            sum03 = fmadd(r03, k003, sum03);

                            sum10 = fmadd(r00, k100, sum10);
                            sum11 = fmadd(r01, k101, sum11);
                            sum12 = fmadd(r02, k102, sum12);
                            sum13 = fmadd(r03, k103, sum13);

                            sum20 = fmadd(r00, k200, sum20);
                            sum21 = fmadd(r01, k201, sum21);
                            sum22 = fmadd(r02, k202, sum22);
                            sum23 = fmadd(r03, k203, sum23);

                            sum30 = fmadd(r00, k300, sum30);
                            sum31 = fmadd(r01, k301, sum31);
                            sum32 = fmadd(r02, k302, sum32);
                            sum33 = fmadd(r03, k303, sum33);

                        }

                        sum00.store(out_0); sum01.store(out_0 + 4); sum02.store(out_0 + 8); sum03.store(out_0 + 12);
                        sum10.store(out_1); sum11.store(out_1 + 4); sum12.store(out_1 + 8); sum13.store(out_1 + 12);
                        sum20.store(out_2); sum21.store(out_2 + 4); sum22.store(out_2 + 8); sum23.store(out_2 + 12);
                        sum30.store(out_3); sum31.store(out_3 + 4); sum32.store(out_3 + 8); sum33.store(out_3 + 12);

                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = remain_outch; c < output_channel; c++)
                {
                    float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const float* kernel_tm_ptr = k_tm.data<float>();
                    const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        float* out_0 = out_tm_0 + i * 16;
                        //float sum_0[16] = { float(0) };

                        float32x4 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f);

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 16;
                            const float* r1 = r0 + tm_c_offset;
                            const float* r2 = r1 + tm_c_offset;
                            const float* r3 = r2 + tm_c_offset;

                            float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);
                            float32x4 r10(r1), r11(r1 + 4), r12(r1 + 8), r13(r1 + 12);
                            float32x4 r20(r2), r21(r2 + 4), r22(r2 + 8), r23(r2 + 12);
                            float32x4 r30(r3), r31(r3 + 4), r32(r3 + 8), r33(r3 + 12);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = k0 + ktm_c_offset;
                            const float* k2 = k1 + ktm_c_offset;
                            const float* k3 = k2 + ktm_c_offset;

                            float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);
                            float32x4 k010(k1), k011(k1 + 4), k012(k1 + 8), k013(k1 + 12);
                            float32x4 k020(k2), k021(k2 + 4), k022(k2 + 8), k023(k2 + 12);
                            float32x4 k030(k3), k031(k3 + 4), k032(k3 + 8), k033(k3 + 12);

                            //sum00 += r00 * k000; sum00 += r10 * k010; sum00 += r20 * k020; sum00 += r30 * k030;
                            //sum01 += r01 * k001; sum01 += r11 * k011; sum01 += r21 * k021; sum01 += r31 * k031;
                            //sum02 += r02 * k002; sum02 += r12 * k012; sum02 += r22 * k022; sum02 += r32 * k032;
                            //sum03 += r03 * k003; sum03 += r13 * k013; sum03 += r23 * k023; sum03 += r33 * k033;

                            sum00 = fmadd(r00, k000, sum00); sum00 = fmadd(r10, k010, sum00); sum00 = fmadd(r20, k020, sum00); sum00 = fmadd(r30, k030, sum00);
                            sum01 = fmadd(r01, k001, sum01); sum01 = fmadd(r11, k011, sum01); sum01 = fmadd(r21, k021, sum01); sum01 = fmadd(r31, k031, sum01);
                            sum02 = fmadd(r02, k002, sum02); sum02 = fmadd(r12, k012, sum02); sum02 = fmadd(r22, k022, sum02); sum02 = fmadd(r32, k032, sum02);
                            sum03 = fmadd(r03, k003, sum03); sum03 = fmadd(r13, k013, sum03); sum03 = fmadd(r23, k023, sum03); sum03 = fmadd(r33, k033, sum03);
                            //for (int k = 0; k < 16; k++)
                            //{
                            //    sum_0[k] += r0[k] * k0[k];
                            //    sum_0[k] += r1[k] * k1[k];
                            //    sum_0[k] += r2[k] * k2[k];
                            //    sum_0[k] += r3[k] * k3[k];
                            //}
                        }

                        for (; q < input_channel; q++)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 16;
                            float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);

                            //sum00 += r00 * k000;
                            //sum01 += r01 * k001;
                            //sum02 += r02 * k002;
                            //sum03 += r03 * k003;

                            sum00 = fmadd(r00, k000, sum00);
                            sum01 = fmadd(r01, k001, sum01);
                            sum02 = fmadd(r02, k002, sum02);
                            sum03 = fmadd(r03, k003, sum03);
                        }

                        sum00.store(out_0); sum01.store(out_0 + 4); sum02.store(out_0 + 8); sum03.store(out_0 + 12);

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            float* out_ptr = output_bordered.data<float>();

            // const float AT[2][4] = {
            //     {1.0f,  1.0f,  1.0f,  0.0f},
            //     {0.0f,  1.0f, -1.0f,  1.0f}
            // }; 

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < output_channel; c++)
                {
                    float* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    float* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        float* out_0 = out_at + i * output_w * 2;
                        float* out_1 = out_0 + output_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            float* out_tile = output_tm_at + (i * col_blocks + j) * 16;

                            float32x4 s0(out_tile), s1(out_tile + 4), s2(out_tile + 8), s3(out_tile + 12);

                            // w = A_T * W
                            float32x4 w0 = s0 + s1 + s2;
                            float32x4 w1 = s1 - s2 + s3;
                            float32x4 w2_tmp(0.f), w3_tmp(0.f);

                            // transpose w to w_t
                            //_MM_TRANSPOSE4_PS(w0.value,w1.value,w2_tmp.value,w3_tmp.value);
                            transposex4x4(w0, w1, w2_tmp, w3_tmp);

                            // Y = A_T * w_t
                            float32x4 o0_tmp = w0 + w1 + w2_tmp;
                            float32x4 o1_tmp = w1 - w2_tmp + w3_tmp;
                            float out_0_tmp[4], out_1_tmp[4];
                            o0_tmp.store(out_0_tmp);
                            o1_tmp.store(out_1_tmp);

                            out_0[0] = out_0_tmp[0];
                            out_0[1] = out_0_tmp[1];
                            out_1[0] = out_1_tmp[0];
                            out_1[1] = out_1_tmp[1];

                            out_0 += 2;
                            out_1 += 2;
                        }
                    }
                }
            }

            inner_cut<float>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }
#endif

        //TO DO:Support threadpool on template,only support float now
        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd23_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out)
        {
            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 1) / 2 * 2;
            output_h = (output_h + 1) / 2 * 2;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<T>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            // const float BT[4][4] = {
            //     {1.0f,  0.0f, -1.0f,  0.0f},
            //     {0.0f,  1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  0.00f, 1.0f}
            // };   

            int w_tm = output_w / 2 * 4;
            int h_tm = output_h / 2 * 4;
            int col_blocks = w_tm / 4;
            int row_blocks = h_tm / 4;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 16 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 16 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const T* src_ptr = input_bordered.data<T>();
            T* dst_ptr = input_tm.data<T>();
            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; c++)
                {
                    const T* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    T* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        const T* r0 = src_at + i * input_padded_w * 2;
                        const T* r1 = r0 + input_padded_w;
                        const T* r2 = r1 + input_padded_w;
                        const T* r3 = r2 + input_padded_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            T d0[4], d1[4], d2[4], d3[4];
                            T w0[4], w1[4], w2[4], w3[4];
                            T t0[4], t1[4], t2[4], t3[4];

                            for (int n = 0; n < 4; n++)
                            {
                                d0[n] = r0[n];
                                d1[n] = r1[n];
                                d2[n] = r2[n];
                                d3[n] = r3[n];
                            }

                            // BT * d * B == (BT * (BT*d)T)T

                            // w = BT * d
                            for (int n = 0; n < 4; n++)
                            {
                                w0[n] = d0[n] - d2[n];
                                w1[n] = d1[n] + d2[n];
                                w2[n] = d2[n] - d1[n];
                                w3[n] = d3[n] - d1[n];
                            }

                            // w to wT
                            {
                                t0[0] = w0[0]; t1[0] = w0[1]; t2[0] = w0[2]; t3[0] = w0[3];
                                t0[1] = w1[0]; t1[1] = w1[1]; t2[1] = w1[2]; t3[1] = w1[3];
                                t0[2] = w2[0]; t1[2] = w2[1]; t2[2] = w2[2]; t3[2] = w2[3];
                                t0[3] = w3[0]; t1[3] = w3[1]; t2[3] = w3[2]; t3[3] = w3[3];
                            }

                            // d = BT * wT 
                            for (int n = 0; n < 4; n++)
                            {
                                d0[n] = t0[n] - t2[n];
                                d1[n] = t1[n] + t2[n];
                                d2[n] = t2[n] - t1[n];
                                d3[n] = t3[n] - t1[n];
                            }

                            for (int n = 0; n < 4; n++)
                            {
                                dst_at[n] = d0[n];
                                dst_at[n + 4] = d1[n];
                                dst_at[n + 8] = d2[n];
                                dst_at[n + 12] = d3[n];
                            }

                            r0 += 2;
                            r1 += 2;
                            r2 += 2;
                            r3 += 2;

                            dst_at += 16;

                        }
                    }
                }

            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 16 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 16;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            T* out_tm_ptr = output_tm.data<T>();
            T* input_tm_ptr = dst_ptr;

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_tm_1 = out_tm_0 + outtm_c_offset;
                    T* out_tm_2 = out_tm_1 + outtm_c_offset;
                    T* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();

                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const T* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const T* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const T* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 16;
                        T* out_1 = out_tm_1 + i * 16;
                        T* out_2 = out_tm_2 + i * 16;
                        T* out_3 = out_tm_3 + i * 16;

                        T sum_0[16] = { T(0) };
                        T sum_1[16] = { T(0) };
                        T sum_2[16] = { T(0) };
                        T sum_3[16] = { T(0) };

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;

                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r1[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r2[k] * k0[k];
                                k0 += 16;
                                sum_0[k] += r3[k] * k0[k];
                                k0 -= 48;

                                sum_1[k] += r0[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r1[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r2[k] * k1[k];
                                k1 += 16;
                                sum_1[k] += r3[k] * k1[k];
                                k1 -= 48;

                                sum_2[k] += r0[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r1[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r2[k] * k2[k];
                                k2 += 16;
                                sum_2[k] += r3[k] * k2[k];
                                k2 -= 48;

                                sum_3[k] += r0[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r1[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r2[k] * k3[k];
                                k3 += 16;
                                sum_3[k] += r3[k] * k3[k];
                                k3 -= 48;
                            }
                        }

                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_1[k] += r0[k] * k1[k];
                                sum_2[k] += r0[k] * k2[k];
                                sum_3[k] += r0[k] * k3[k];
                            }
                        }

                        for (int k = 0; k < 16; k++)
                        {
                            out_0[k] = sum_0[k];
                            out_1[k] = sum_1[k];
                            out_2[k] = sum_2[k];
                            out_3[k] = sum_3[k];
                        }

                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = remain_outch; c < output_channel; c++)
                {
                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();
                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 16;
                        T sum_0[16] = { T(0) };

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = k0 + ktm_c_offset;
                            const T* k2 = k1 + ktm_c_offset;
                            const T* k3 = k2 + ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_0[k] += r1[k] * k1[k];
                                sum_0[k] += r2[k] * k2[k];
                                sum_0[k] += r3[k] * k3[k];
                            }
                        }

                        for (; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 16;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;

                            for (int k = 0; k < 16; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                            }
                        }

                        for (int k = 0; k < 16; k++)
                        {
                            out_0[k] = sum_0[k];
                        }

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            T* out_ptr = output_bordered.data<T>();

            // const float AT[2][4] = {
            //     {1.0f,  1.0f,  1.0f,  0.0f},
            //     {0.0f,  1.0f, -1.0f,  1.0f}
            // }; 

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < output_channel; c++)
                {
                    T* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    for (int i = 0; i < col_blocks; i++)
                    {
                        T* out_0 = out_at + i * output_w * 2;
                        T* out_1 = out_0 + output_w;

                        for (int j = 0; j < row_blocks; j++)
                        {
                            T* out_tile = output_tm_at + (i * col_blocks + j) * 16;

                            T s0[4], s1[4], s2[4], s3[4];
                            T w0[4], w1[4];
                            T d0[2], d1[2], d2[2], d3[2];
                            T o0[2], o1[2];

                            for (int n = 0; n < 4; n++)
                            {
                                s0[n] = out_tile[n];
                                s1[n] = out_tile[n + 4];
                                s2[n] = out_tile[n + 8];
                                s3[n] = out_tile[n + 12];
                            }
                            // w = A_T * W
                            for (int n = 0; n < 4; n++)
                            {
                                w0[n] = s0[n] + s1[n] + s2[n];
                                w1[n] = s1[n] - s2[n] + s3[n];
                            }
                            // transpose w to w_t
                            {
                                d0[0] = w0[0]; d0[1] = w1[0];
                                d1[0] = w0[1]; d1[1] = w1[1];
                                d2[0] = w0[2]; d2[1] = w1[2];
                                d3[0] = w0[3]; d3[1] = w1[3];
                            }
                            // Y = A_T * w_t
                            for (int n = 0; n < 2; n++)
                            {
                                o0[n] = d0[n] + d1[n] + d2[n];
                                o1[n] = d1[n] - d2[n] + d3[n];
                            }

                            out_0[0] = o0[0];
                            out_0[1] = o0[1];
                            out_1[0] = o1[0];
                            out_1[1] = o1[1];

                            out_0 += 2;
                            out_1 += 2;
                        }
                    }
                }
            }

            inner_cut<T>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);
        }

        template<>
        void Conv2dAlgorithm<float>::conv3x3_winograd23_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out)
        {
            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 1) / 2 * 2;
            output_h = (output_h + 1) / 2 * 2;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<float>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            // const float BT[4][4] = {
            //     {1.0f,  0.0f, -1.0f,  0.0f},
            //     {0.0f,  1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  1.00f, 0.0f},
            //     {0.0f, -1.0f,  0.00f, 1.0f}
            // };   

            int w_tm = output_w / 2 * 4;
            int h_tm = output_h / 2 * 4;
            int col_blocks = w_tm / 4;
            int row_blocks = h_tm / 4;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 16 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 16 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            auto thread_pool = ctx::lite::ptr<ThreadPool>();

            const float* src_ptr = input_bordered.data<float>();
            float* dst_ptr = input_tm.data<float>();
            if (thread_pool == nullptr || thread_pool->size() <= 1)
            {
                for (int n = 0; n < num; n++)
                {
                    for (int c = 0; c < input_channel; c++)
                    {
                        const float* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                        float* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                        for (int i = 0; i < col_blocks; i++)
                        {
                            const float* r0 = src_at + i * input_padded_w * 2;
                            const float* r1 = r0 + input_padded_w;
                            const float* r2 = r1 + input_padded_w;
                            const float* r3 = r2 + input_padded_w;

                            for (int j = 0; j < row_blocks; j++)
                            {
                                float32x4 dst0(dst_at);
                                float32x4 dst1(dst_at + 4);
                                float32x4 dst2(dst_at + 8);
                                float32x4 dst3(dst_at + 12);

                                float32x4 d0(r0), d1(r1), d2(r2), d3(r3);

                                // BT * d * B == (BT * (BT*d)T)T
                                // w = BT * d
                                float32x4 w0 = d0 - d2;
                                float32x4 w1 = d1 + d2;
                                float32x4 w2 = d2 - d1;
                                float32x4 w3 = d3 - d1;

                                // w to wT
                                //_MM_TRANSPOSE4_PS(w0.value,w1.value,w2.value,w3.value);
                                transposex4x4(w0, w1, w2, w3);
                                // d = BT * wT
                                d0 = w0 - w2;
                                d1 = w1 + w2;
                                d2 = w2 - w1;
                                d3 = w3 - w1;

                                d0.store(dst_at);
                                d1.store(dst_at + 4);
                                d2.store(dst_at + 8);
                                d3.store(dst_at + 12);

                                r0 += 2;
                                r1 += 2;
                                r2 += 2;
                                r3 += 2;

                                dst_at += 16;

                            }
                        }
                    }
                }
            }
            else
            {
                for (int n = 0; n < num; n++)
                {
                    auto bins = split_bins(0, input_channel, (int)thread_pool->size());
                    for (auto &bin : bins)
                    {
                        thread_pool->run([&, n, src_ptr, dst_ptr, bin](int) {
                            const float* src_at = src_ptr + n * bordered_num_offset + bin.first * bordered_c_offset;
                            float* dst_at = dst_ptr + n * tm_num_offset + bin.first * tm_c_offset;

                            for (int c = bin.first; c < bin.second; c++)
                            {
                                for (int i = 0; i < col_blocks; i++)
                                {
                                    const float* r0 = src_at + i * input_padded_w * 2;
                                    const float* r1 = r0 + input_padded_w;
                                    const float* r2 = r1 + input_padded_w;
                                    const float* r3 = r2 + input_padded_w;

                                    for (int j = 0; j < row_blocks; j++)
                                    {
                                        float32x4 dst0(dst_at);
                                        float32x4 dst1(dst_at + 4);
                                        float32x4 dst2(dst_at + 8);
                                        float32x4 dst3(dst_at + 12);

                                        float32x4 d0(r0), d1(r1), d2(r2), d3(r3);

                                        // BT * d * B == (BT * (BT*d)T)T
                                        // w = BT * d
                                        float32x4 w0 = d0 - d2;
                                        float32x4 w1 = d1 + d2;
                                        float32x4 w2 = d2 - d1;
                                        float32x4 w3 = d3 - d1;

                                        // w to wT
                                        //_MM_TRANSPOSE4_PS(w0.value,w1.value,w2.value,w3.value);
                                        transposex4x4(w0, w1, w2, w3);
                                        // d = BT * wT
                                        d0 = w0 - w2;
                                        d1 = w1 + w2;
                                        d2 = w2 - w1;
                                        d3 = w3 - w1;

                                        d0.store(dst_at);
                                        d1.store(dst_at + 4);
                                        d2.store(dst_at + 8);
                                        d3.store(dst_at + 12);

                                        r0 += 2;
                                        r1 += 2;
                                        r2 += 2;
                                        r3 += 2;

                                        dst_at += 16;

                                    }
                                }
                                src_at += bordered_c_offset;
                            }
                        });

                    }
                }
                thread_pool->join();
            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 16 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 16;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            float* out_tm_ptr = output_tm.data<float>();
            float* input_tm_ptr = dst_ptr;
            const float* kernel_tm_ptr = k_tm.data<float>();

            for (int n = 0; n < num; n++)
            {
                if (thread_pool == nullptr || thread_pool->size() <= 1)
                {
                    for (int cc = 0; cc < outch; cc++)
                    {
                        int c = cc * 4;

                        float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                        float* out_tm_1 = out_tm_0 + outtm_c_offset;
                        float* out_tm_2 = out_tm_1 + outtm_c_offset;
                        float* out_tm_3 = out_tm_2 + outtm_c_offset;

                        const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                        const float* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                        const float* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                        const float* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                        for (int i = 0; i < num_blocks; i++)
                        {
                            float* out_0 = out_tm_0 + i * 16;
                            float* out_1 = out_tm_1 + i * 16;
                            float* out_2 = out_tm_2 + i * 16;
                            float* out_3 = out_tm_3 + i * 16;

                            float32x4 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f);
                            float32x4 sum10(0.f), sum11(0.f), sum12(0.f), sum13(0.f);
                            float32x4 sum20(0.f), sum21(0.f), sum22(0.f), sum23(0.f);
                            float32x4 sum30(0.f), sum31(0.f), sum32(0.f), sum33(0.f);

                            int inputch = input_channel >> 2;
                            int remain_inputch = inputch << 2;

                            //#pragma omp parallel for num_threads(omp_get_max_threads())
                            for (int qq = 0; qq < inputch; qq++)
                            {
                                int q = qq * 4;
                                const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                const float* r0 = input_tm_at + i * 16;
                                const float* r1 = r0 + tm_c_offset;
                                const float* r2 = r1 + tm_c_offset;
                                const float* r3 = r2 + tm_c_offset;
                                float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);
                                float32x4 r10(r1), r11(r1 + 4), r12(r1 + 8), r13(r1 + 12);
                                float32x4 r20(r2), r21(r2 + 4), r22(r2 + 8), r23(r2 + 12);
                                float32x4 r30(r3), r31(r3 + 4), r32(r3 + 8), r33(r3 + 12);

                                const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                const float* k3 = kernel_tm_3 + q * ktm_c_offset;
                                float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);
                                float32x4 k010(k0 + 16), k011(k0 + 20), k012(k0 + 24), k013(k0 + 28);
                                float32x4 k020(k0 + 32), k021(k0 + 36), k022(k0 + 40), k023(k0 + 44);
                                float32x4 k030(k0 + 48), k031(k0 + 52), k032(k0 + 56), k033(k0 + 60);

                                float32x4 k100(k1), k101(k1 + 4), k102(k1 + 8), k103(k1 + 12);
                                float32x4 k110(k1 + 16), k111(k1 + 20), k112(k1 + 24), k113(k1 + 28);
                                float32x4 k120(k1 + 32), k121(k1 + 36), k122(k1 + 40), k123(k1 + 44);
                                float32x4 k130(k1 + 48), k131(k1 + 52), k132(k1 + 56), k133(k1 + 60);

                                float32x4 k200(k2), k201(k2 + 4), k202(k2 + 8), k203(k2 + 12);
                                float32x4 k210(k2 + 16), k211(k2 + 20), k212(k2 + 24), k213(k2 + 28);
                                float32x4 k220(k2 + 32), k221(k2 + 36), k222(k2 + 40), k223(k2 + 44);
                                float32x4 k230(k2 + 48), k231(k2 + 52), k232(k2 + 56), k233(k2 + 60);

                                float32x4 k300(k3), k301(k3 + 4), k302(k3 + 8), k303(k3 + 12);
                                float32x4 k310(k3 + 16), k311(k3 + 20), k312(k3 + 24), k313(k3 + 28);
                                float32x4 k320(k3 + 32), k321(k3 + 36), k322(k3 + 40), k323(k3 + 44);
                                float32x4 k330(k3 + 48), k331(k3 + 52), k332(k3 + 56), k333(k3 + 60);

                                sum00 += r00 * k000; sum00 += r10 * k010; sum00 += r20 * k020; sum00 += r30 * k030;
                                sum01 += r01 * k001; sum01 += r11 * k011; sum01 += r21 * k021; sum01 += r31 * k031;
                                sum02 += r02 * k002; sum02 += r12 * k012; sum02 += r22 * k022; sum02 += r32 * k032;
                                sum03 += r03 * k003; sum03 += r13 * k013; sum03 += r23 * k023; sum03 += r33 * k033;

                                sum10 += r00 * k100; sum10 += r10 * k110; sum10 += r20 * k120; sum10 += r30 * k130;
                                sum11 += r01 * k101; sum11 += r11 * k111; sum11 += r21 * k121; sum11 += r31 * k131;
                                sum12 += r02 * k102; sum12 += r12 * k112; sum12 += r22 * k122; sum12 += r32 * k132;
                                sum13 += r03 * k103; sum13 += r13 * k113; sum13 += r23 * k123; sum13 += r33 * k133;

                                sum20 += r00 * k200; sum20 += r10 * k210; sum20 += r20 * k220; sum20 += r30 * k230;
                                sum21 += r01 * k201; sum21 += r11 * k211; sum21 += r21 * k221; sum21 += r31 * k231;
                                sum22 += r02 * k202; sum22 += r12 * k212; sum22 += r22 * k222; sum22 += r32 * k232;
                                sum23 += r03 * k203; sum23 += r13 * k213; sum23 += r23 * k223; sum23 += r33 * k233;

                                sum30 += r00 * k300; sum30 += r10 * k310; sum30 += r20 * k320; sum30 += r30 * k330;
                                sum31 += r01 * k301; sum31 += r11 * k311; sum31 += r21 * k321; sum31 += r31 * k331;
                                sum32 += r02 * k302; sum32 += r12 * k312; sum32 += r22 * k322; sum32 += r32 * k332;
                                sum33 += r03 * k303; sum33 += r13 * k313; sum33 += r23 * k323; sum33 += r33 * k333;

                            }

                            for (int q = remain_inputch; q < input_channel; q++)
                            {
                                const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                const float* r0 = input_tm_at + i * 16;
                                float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);

                                const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                const float* k3 = kernel_tm_3 + q * ktm_c_offset;

                                float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);

                                float32x4 k100(k1), k101(k1 + 4), k102(k1 + 8), k103(k1 + 12);

                                float32x4 k200(k2), k201(k2 + 4), k202(k2 + 8), k203(k2 + 12);

                                float32x4 k300(k3), k301(k3 + 4), k302(k3 + 8), k303(k3 + 12);

                                sum00 += r00 * k000;
                                sum01 += r01 * k001;
                                sum02 += r02 * k002;
                                sum03 += r03 * k003;

                                sum10 += r00 * k100;
                                sum11 += r01 * k101;
                                sum12 += r02 * k102;
                                sum13 += r03 * k103;

                                sum20 += r00 * k200;
                                sum21 += r01 * k201;
                                sum22 += r02 * k202;
                                sum23 += r03 * k203;

                                sum30 += r00 * k300;
                                sum31 += r01 * k301;
                                sum32 += r02 * k302;
                                sum33 += r03 * k303;

                            }

                            sum00.store(out_0); sum01.store(out_0 + 4); sum02.store(out_0 + 8); sum03.store(out_0 + 12);
                            sum10.store(out_1); sum11.store(out_1 + 4); sum12.store(out_1 + 8); sum13.store(out_1 + 12);
                            sum20.store(out_2); sum21.store(out_2 + 4); sum22.store(out_2 + 8); sum23.store(out_2 + 12);
                            sum30.store(out_3); sum31.store(out_3 + 4); sum32.store(out_3 + 8); sum33.store(out_3 + 12);

                        }
                    }
                }
                else
                {
                    auto bins = split_bins(0, outch, (int)thread_pool->size());
                    for (auto &bin : bins)
                    {
                        thread_pool->run([&, n, input_tm_ptr, out_tm_ptr, kernel_tm_ptr, bin](int) {
                            float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + 4 * bin.first * outtm_c_offset;

                            const float* kernel_tm_0 = kernel_tm_ptr + 4 * bin.first * ktm_n_offset;

                            for (int c = bin.first; c < bin.second; c++)
                            {
                                float* out_tm_1 = out_tm_0 + outtm_c_offset;
                                float* out_tm_2 = out_tm_1 + outtm_c_offset;
                                float* out_tm_3 = out_tm_2 + outtm_c_offset;

                                const float* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                                const float* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                                const float* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                                for (int i = 0; i < num_blocks; i++)
                                {
                                    float* out_0 = out_tm_0 + i * 16;
                                    float* out_1 = out_tm_1 + i * 16;
                                    float* out_2 = out_tm_2 + i * 16;
                                    float* out_3 = out_tm_3 + i * 16;

                                    float32x4 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f);
                                    float32x4 sum10(0.f), sum11(0.f), sum12(0.f), sum13(0.f);
                                    float32x4 sum20(0.f), sum21(0.f), sum22(0.f), sum23(0.f);
                                    float32x4 sum30(0.f), sum31(0.f), sum32(0.f), sum33(0.f);

                                    int inputch = input_channel >> 2;
                                    int remain_inputch = inputch << 2;

                                    for (int qq = 0; qq < inputch; qq++)
                                    {
                                        int q = qq * 4;
                                        const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                        const float* r0 = input_tm_at + i * 16;
                                        const float* r1 = r0 + tm_c_offset;
                                        const float* r2 = r1 + tm_c_offset;
                                        const float* r3 = r2 + tm_c_offset;
                                        float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);
                                        float32x4 r10(r1), r11(r1 + 4), r12(r1 + 8), r13(r1 + 12);
                                        float32x4 r20(r2), r21(r2 + 4), r22(r2 + 8), r23(r2 + 12);
                                        float32x4 r30(r3), r31(r3 + 4), r32(r3 + 8), r33(r3 + 12);

                                        const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                        const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                        const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                        const float* k3 = kernel_tm_3 + q * ktm_c_offset;
                                        float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);
                                        float32x4 k010(k0 + 16), k011(k0 + 20), k012(k0 + 24), k013(k0 + 28);
                                        float32x4 k020(k0 + 32), k021(k0 + 36), k022(k0 + 40), k023(k0 + 44);
                                        float32x4 k030(k0 + 48), k031(k0 + 52), k032(k0 + 56), k033(k0 + 60);

                                        float32x4 k100(k1), k101(k1 + 4), k102(k1 + 8), k103(k1 + 12);
                                        float32x4 k110(k1 + 16), k111(k1 + 20), k112(k1 + 24), k113(k1 + 28);
                                        float32x4 k120(k1 + 32), k121(k1 + 36), k122(k1 + 40), k123(k1 + 44);
                                        float32x4 k130(k1 + 48), k131(k1 + 52), k132(k1 + 56), k133(k1 + 60);

                                        float32x4 k200(k2), k201(k2 + 4), k202(k2 + 8), k203(k2 + 12);
                                        float32x4 k210(k2 + 16), k211(k2 + 20), k212(k2 + 24), k213(k2 + 28);
                                        float32x4 k220(k2 + 32), k221(k2 + 36), k222(k2 + 40), k223(k2 + 44);
                                        float32x4 k230(k2 + 48), k231(k2 + 52), k232(k2 + 56), k233(k2 + 60);

                                        float32x4 k300(k3), k301(k3 + 4), k302(k3 + 8), k303(k3 + 12);
                                        float32x4 k310(k3 + 16), k311(k3 + 20), k312(k3 + 24), k313(k3 + 28);
                                        float32x4 k320(k3 + 32), k321(k3 + 36), k322(k3 + 40), k323(k3 + 44);
                                        float32x4 k330(k3 + 48), k331(k3 + 52), k332(k3 + 56), k333(k3 + 60);

                                        sum00 += r00 * k000; sum00 += r10 * k010; sum00 += r20 * k020; sum00 += r30 * k030;
                                        sum01 += r01 * k001; sum01 += r11 * k011; sum01 += r21 * k021; sum01 += r31 * k031;
                                        sum02 += r02 * k002; sum02 += r12 * k012; sum02 += r22 * k022; sum02 += r32 * k032;
                                        sum03 += r03 * k003; sum03 += r13 * k013; sum03 += r23 * k023; sum03 += r33 * k033;

                                        sum10 += r00 * k100; sum10 += r10 * k110; sum10 += r20 * k120; sum10 += r30 * k130;
                                        sum11 += r01 * k101; sum11 += r11 * k111; sum11 += r21 * k121; sum11 += r31 * k131;
                                        sum12 += r02 * k102; sum12 += r12 * k112; sum12 += r22 * k122; sum12 += r32 * k132;
                                        sum13 += r03 * k103; sum13 += r13 * k113; sum13 += r23 * k123; sum13 += r33 * k133;

                                        sum20 += r00 * k200; sum20 += r10 * k210; sum20 += r20 * k220; sum20 += r30 * k230;
                                        sum21 += r01 * k201; sum21 += r11 * k211; sum21 += r21 * k221; sum21 += r31 * k231;
                                        sum22 += r02 * k202; sum22 += r12 * k212; sum22 += r22 * k222; sum22 += r32 * k232;
                                        sum23 += r03 * k203; sum23 += r13 * k213; sum23 += r23 * k223; sum23 += r33 * k233;

                                        sum30 += r00 * k300; sum30 += r10 * k310; sum30 += r20 * k320; sum30 += r30 * k330;
                                        sum31 += r01 * k301; sum31 += r11 * k311; sum31 += r21 * k321; sum31 += r31 * k331;
                                        sum32 += r02 * k302; sum32 += r12 * k312; sum32 += r22 * k322; sum32 += r32 * k332;
                                        sum33 += r03 * k303; sum33 += r13 * k313; sum33 += r23 * k323; sum33 += r33 * k333;

                                    }

                                    for (int q = remain_inputch; q < input_channel; q++)
                                    {
                                        const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                        const float* r0 = input_tm_at + i * 16;
                                        float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);

                                        const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                        const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                        const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                        const float* k3 = kernel_tm_3 + q * ktm_c_offset;

                                        float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);

                                        float32x4 k100(k1), k101(k1 + 4), k102(k1 + 8), k103(k1 + 12);

                                        float32x4 k200(k2), k201(k2 + 4), k202(k2 + 8), k203(k2 + 12);

                                        float32x4 k300(k3), k301(k3 + 4), k302(k3 + 8), k303(k3 + 12);

                                        sum00 += r00 * k000;
                                        sum01 += r01 * k001;
                                        sum02 += r02 * k002;
                                        sum03 += r03 * k003;

                                        sum10 += r00 * k100;
                                        sum11 += r01 * k101;
                                        sum12 += r02 * k102;
                                        sum13 += r03 * k103;

                                        sum20 += r00 * k200;
                                        sum21 += r01 * k201;
                                        sum22 += r02 * k202;
                                        sum23 += r03 * k203;

                                        sum30 += r00 * k300;
                                        sum31 += r01 * k301;
                                        sum32 += r02 * k302;
                                        sum33 += r03 * k303;

                                    }

                                    sum00.store(out_0); sum01.store(out_0 + 4); sum02.store(out_0 + 8); sum03.store(out_0 + 12);
                                    sum10.store(out_1); sum11.store(out_1 + 4); sum12.store(out_1 + 8); sum13.store(out_1 + 12);
                                    sum20.store(out_2); sum21.store(out_2 + 4); sum22.store(out_2 + 8); sum23.store(out_2 + 12);
                                    sum30.store(out_3); sum31.store(out_3 + 4); sum32.store(out_3 + 8); sum33.store(out_3 + 12);

                                }
                                out_tm_0 += (4 * outtm_c_offset);
                                kernel_tm_0 += (4 * ktm_n_offset);
                            }
                        });
                    }
                    thread_pool->join();
                }

                for (int c = remain_outch; c < output_channel; c++)
                {
                    float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const float* kernel_tm_ptr = k_tm.data<float>();
                    const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        float* out_0 = out_tm_0 + i * 16;
                        //float sum_0[16] = { float(0) };

                        float32x4 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f);

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 16;
                            const float* r1 = r0 + tm_c_offset;
                            const float* r2 = r1 + tm_c_offset;
                            const float* r3 = r2 + tm_c_offset;

                            float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);
                            float32x4 r10(r1), r11(r1 + 4), r12(r1 + 8), r13(r1 + 12);
                            float32x4 r20(r2), r21(r2 + 4), r22(r2 + 8), r23(r2 + 12);
                            float32x4 r30(r3), r31(r3 + 4), r32(r3 + 8), r33(r3 + 12);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = k0 + ktm_c_offset;
                            const float* k2 = k1 + ktm_c_offset;
                            const float* k3 = k2 + ktm_c_offset;

                            float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);
                            float32x4 k010(k1), k011(k1 + 4), k012(k1 + 8), k013(k1 + 12);
                            float32x4 k020(k2), k021(k2 + 4), k022(k2 + 8), k023(k2 + 12);
                            float32x4 k030(k3), k031(k3 + 4), k032(k3 + 8), k033(k3 + 12);

                            sum00 += r00 * k000; sum00 += r10 * k010; sum00 += r20 * k020; sum00 += r30 * k030;
                            sum01 += r01 * k001; sum01 += r11 * k011; sum01 += r21 * k021; sum01 += r31 * k031;
                            sum02 += r02 * k002; sum02 += r12 * k012; sum02 += r22 * k022; sum02 += r32 * k032;
                            sum03 += r03 * k003; sum03 += r13 * k013; sum03 += r23 * k023; sum03 += r33 * k033;

                            //for (int k = 0; k < 16; k++)
                            //{
                            //    sum_0[k] += r0[k] * k0[k];
                            //    sum_0[k] += r1[k] * k1[k];
                            //    sum_0[k] += r2[k] * k2[k];
                            //    sum_0[k] += r3[k] * k3[k];
                            //}
                        }

                        for (; q < input_channel; q++)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 16;
                            float32x4 r00(r0), r01(r0 + 4), r02(r0 + 8), r03(r0 + 12);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            float32x4 k000(k0), k001(k0 + 4), k002(k0 + 8), k003(k0 + 12);

                            sum00 += r00 * k000;
                            sum01 += r01 * k001;
                            sum02 += r02 * k002;
                            sum03 += r03 * k003;
                        }

                        sum00.store(out_0); sum01.store(out_0 + 4); sum02.store(out_0 + 8); sum03.store(out_0 + 12);

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            float* out_ptr = output_bordered.data<float>();

            // const float AT[2][4] = {
            //     {1.0f,  1.0f,  1.0f,  0.0f},
            //     {0.0f,  1.0f, -1.0f,  1.0f}
            // }; 

            for (int n = 0; n < num; n++)
            {
                if (thread_pool == nullptr || thread_pool->size() <= 1)
                {
                    for (int c = 0; c < output_channel; c++)
                    {
                        float* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                        float* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                        for (int i = 0; i < col_blocks; i++)
                        {
                            float* out_0 = out_at + i * output_w * 2;
                            float* out_1 = out_0 + output_w;

                            for (int j = 0; j < row_blocks; j++)
                            {
                                float* out_tile = output_tm_at + (i * col_blocks + j) * 16;

                                float32x4 s0(out_tile), s1(out_tile + 4), s2(out_tile + 8), s3(out_tile + 12);

                                // w = A_T * W
                                float32x4 w0 = s0 + s1 + s2;
                                float32x4 w1 = s1 - s2 + s3;
                                float32x4 w2_tmp(0.f), w3_tmp(0.f);

                                // transpose w to w_t
                                //_MM_TRANSPOSE4_PS(w0.value,w1.value,w2_tmp.value,w3_tmp.value);
                                transposex4x4(w0, w1, w2_tmp, w3_tmp);

                                // Y = A_T * w_t
                                float32x4 o0_tmp = w0 + w1 + w2_tmp;
                                float32x4 o1_tmp = w1 - w2_tmp + w3_tmp;
                                float out_0_tmp[4], out_1_tmp[4];
                                o0_tmp.store(out_0_tmp);
                                o1_tmp.store(out_1_tmp);

                                out_0[0] = out_0_tmp[0];
                                out_0[1] = out_0_tmp[1];
                                out_1[0] = out_1_tmp[0];
                                out_1[1] = out_1_tmp[1];

                                out_0 += 2;
                                out_1 += 2;
                            }
                        }
                    }
                }
                else
                {
                    auto bins = split_bins(0, output_channel, (int)thread_pool->size());
                    for (auto &bin : bins)
                    {
                        thread_pool->run([&, n, out_tm_ptr, out_ptr, bin](int) {
                            float* output_tm_at = out_tm_ptr + n * outtm_n_offset + bin.first * outtm_c_offset;
                            float* out_at = out_ptr + n * outbo_n_offset + bin.first * outbo_c_offset;

                            for (int c = bin.first; c < bin.second; c++)
                            {
                                for (int i = 0; i < col_blocks; i++)
                                {
                                    float* out_0 = out_at + i * output_w * 2;
                                    float* out_1 = out_0 + output_w;

                                    for (int j = 0; j < row_blocks; j++)
                                    {
                                        float* out_tile = output_tm_at + (i * col_blocks + j) * 16;

                                        float32x4 s0(out_tile), s1(out_tile + 4), s2(out_tile + 8), s3(out_tile + 12);

                                        // w = A_T * W
                                        float32x4 w0 = s0 + s1 + s2;
                                        float32x4 w1 = s1 - s2 + s3;
                                        float32x4 w2_tmp(0.f), w3_tmp(0.f);

                                        // transpose w to w_t
                                        //_MM_TRANSPOSE4_PS(w0.value,w1.value,w2_tmp.value,w3_tmp.value);
                                        transposex4x4(w0, w1, w2_tmp, w3_tmp);

                                        // Y = A_T * w_t
                                        float32x4 o0_tmp = w0 + w1 + w2_tmp;
                                        float32x4 o1_tmp = w1 - w2_tmp + w3_tmp;
                                        float out_0_tmp[4], out_1_tmp[4];
                                        o0_tmp.store(out_0_tmp);
                                        o1_tmp.store(out_1_tmp);

                                        out_0[0] = out_0_tmp[0];
                                        out_0[1] = out_0_tmp[1];
                                        out_1[0] = out_1_tmp[0];
                                        out_1[1] = out_1_tmp[1];

                                        out_0 += 2;
                                        out_1 += 2;
                                    }
                                }
                                output_tm_at += outtm_c_offset;
                                out_at += outbo_c_offset;
                            }
                        });
                    }
                    thread_pool->join();
                }
            }

            inner_cut<float>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd63(const Tensor &x, const Tensor &k_tm, Tensor &out) {

            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 5) / 6 * 6;
            output_h = (output_h + 5) / 6 * 6;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<T>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            //const float BT[8][8] = {
            //    {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
            //
            //    {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
            //    {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
            //    {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
            //    {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
            //
            //    {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
            //};

            int w_tm = output_w / 6 * 8;
            int h_tm = output_h / 6 * 8;
            int col_blocks = w_tm / 8;
            int row_blocks = h_tm / 8;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 64 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 64 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const T* src_ptr = input_bordered.data<T>();
            T* dst_ptr = input_tm.data<T>();
            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; c++)
                {
                    const T* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    T* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    T tmp[8][8];//save (d*B)T
                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const T* r0 = src_at + i * input_padded_w * 6 + j * 6;

                            for (int m = 0; m < 8; m++)
                            {
                                tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                                tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                                T tmp12_a = (r0[2] + r0[6] - r0[4] * 4.25f);
                                T tmp12_b = (r0[1] + r0[5] - r0[3] * 4.25f);

                                tmp[1][m] = tmp12_a + tmp12_b;
                                tmp[2][m] = tmp12_a - tmp12_b;

                                T tmp34_a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                                T tmp34_b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

                                tmp[3][m] = tmp34_a + tmp34_b;
                                tmp[4][m] = tmp34_a - tmp34_b;

                                T tmp56_a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                                T tmp56_b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                                tmp[5][m] = tmp56_a + tmp56_b;
                                tmp[6][m] = tmp56_a - tmp56_b;

                                r0 += input_padded_w;
                            }

                            T* d0 = dst_at + (i * col_blocks + j) * 64;

                            T* d1 = d0 + 1;
                            T* d2 = d1 + 1;
                            T* d3 = d2 + 1;
                            T* d4 = d3 + 1;
                            T* d5 = d4 + 1;
                            T* d6 = d5 + 1;
                            T* d7 = d6 + 1;

                            //(d*B)T * B == (BT*d*B)T == VT
                            for (int m = 0; m < 8; m++)
                            {
                                const T* tmp0 = tmp[m];

                                d0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * T(5.25);
                                d7[0] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * T(5.25);

                                T tmp12_a = (tmp0[2] + tmp0[6] - tmp0[4] * T(4.25));
                                T tmp12_b = (tmp0[1] - tmp0[3] * T(4.25) + tmp0[5]);

                                d1[0] = tmp12_a + tmp12_b;
                                d2[0] = tmp12_a - tmp12_b;

                                T tmp34_a = (tmp0[6] + tmp0[2] * T(0.25) - tmp0[4] * T(1.25));
                                T tmp34_b = (tmp0[1] * T(0.5) - tmp0[3] * T(2.5) + tmp0[5] * T(2));

                                d3[0] = tmp34_a + tmp34_b;
                                d4[0] = tmp34_a - tmp34_b;

                                T tmp56_a = (tmp0[6] + (tmp0[2] - tmp0[4] * T(1.25)) * T(4));
                                T tmp56_b = (tmp0[1] * T(2) - tmp0[3] * T(2.5) + tmp0[5] * T(0.5));

                                d5[0] = tmp56_a + tmp56_b;
                                d6[0] = tmp56_a - tmp56_b;

                                d0 += 8;
                                d1 += 8;
                                d2 += 8;
                                d3 += 8;
                                d4 += 8;
                                d5 += 8;
                                d6 += 8;
                                d7 += 8;

                            }
                        }
                    }
                }

            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 64 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 64;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            T* out_tm_ptr = output_tm.data<T>();
            T* input_tm_ptr = dst_ptr;

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_tm_1 = out_tm_0 + outtm_c_offset;
                    T* out_tm_2 = out_tm_1 + outtm_c_offset;
                    T* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();

                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const T* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const T* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const T* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 64;
                        T* out_1 = out_tm_1 + i * 64;
                        T* out_2 = out_tm_2 + i * 64;
                        T* out_3 = out_tm_3 + i * 64;

                        T sum_0[64] = { T(0) };
                        T sum_1[64] = { T(0) };
                        T sum_2[64] = { T(0) };
                        T sum_3[64] = { T(0) };

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;

                        //#pragma omp parallel for num_threads(omp_get_max_threads())
                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r1[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r2[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r3[k] * k0[k];
                                k0 -= 192;

                                sum_1[k] += r0[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r1[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r2[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r3[k] * k1[k];
                                k1 -= 192;

                                sum_2[k] += r0[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r1[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r2[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r3[k] * k2[k];
                                k2 -= 192;

                                sum_3[k] += r0[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r1[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r2[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r3[k] * k3[k];
                                k3 -= 192;
                            }
                        }
                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_1[k] += r0[k] * k1[k];
                                sum_2[k] += r0[k] * k2[k];
                                sum_3[k] += r0[k] * k3[k];
                            }
                        }

                        for (int k = 0; k < 64; k++)
                        {
                            out_0[k] = sum_0[k];
                            out_1[k] = sum_1[k];
                            out_2[k] = sum_2[k];
                            out_3[k] = sum_3[k];
                        }

                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = remain_outch; c < output_channel; c++)
                {
                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();
                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 64;
                        T sum_0[64] = { T(0) };

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;
                            // const T* r1 = r0 + tm_c_offset;
                            // const T* r2 = r1 + tm_c_offset;
                            // const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = k0 + ktm_c_offset;
                            const T* k2 = k1 + ktm_c_offset;
                            const T* k3 = k2 + ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_0[k] += r0[k] * k1[k];
                                sum_0[k] += r0[k] * k2[k];
                                sum_0[k] += r0[k] * k3[k];
                            }
                        }

                        for (; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                            }
                        }

                        for (int k = 0; k < 64; k++)
                        {
                            out_0[k] = sum_0[k];
                        }

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            T* out_ptr = output_bordered.data<T>();

            //const float AT[6][8] = {
            //    {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
            //};

            // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
            // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
            // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
            // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
            // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
            // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

            //reuse (r1 + r2) (r1 - r2) (r3 + r4) (r3 - r4) (r5 + r6) (r5 - r6)

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < output_channel; c++)
                {
                    T* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    T tmp[6][8]; //(WT*A)T == AT*W
                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const T* w0 = output_tm_at + (i * col_blocks + j) * 64;

                            for (int m = 0; m < 8; m++)
                            {
                                T tmp1add2 = w0[1] + w0[2];
                                T tmp1sub2 = w0[1] - w0[2];
                                T tmp3add4 = w0[3] + w0[4];
                                T tmp3sub4 = w0[3] - w0[4];
                                T tmp5add6 = w0[5] + w0[6];
                                T tmp5sub6 = w0[5] - w0[6];

                                tmp[0][m] = w0[0] + tmp1add2 + tmp3add4 + tmp5add6 * 32;
                                tmp[1][m] = tmp1sub2 + tmp3sub4 * 2 + tmp5sub6 * 16;
                                tmp[2][m] = tmp1add2 + tmp3add4 * 4 + tmp5add6 * 8;
                                tmp[3][m] = tmp1sub2 + tmp3sub4 * 8 + tmp5sub6 * 4;
                                tmp[4][m] = tmp1add2 + tmp3add4 * 16 + tmp5add6 * 2;
                                tmp[5][m] = tmp1sub2 + tmp3sub4 * 32 + tmp5sub6 + w0[7];

                                //tmp[0][m] = w0[0] + (w0[1] + w0[2]) + (w0[3] + w0[4]) + (w0[5] + w0[6]) * 32;
                                //tmp[1][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 2 + (w0[5] - w0[6]) * 16;
                                //tmp[2][m] = (w0[1] + w0[2]) + (w0[3] + w0[4]) * 4 + (w0[5] + w0[6]) * 8;
                                //tmp[3][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 8 + (w0[5] - w0[6]) * 4;
                                //tmp[4][m] = (w0[1] + w0[2]) + (w0[3] + w0[4]) * 16 + (w0[5] + w0[6]) * 2;
                                //tmp[5][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 32 + (w0[5] - w0[6]) + w0[7];

                                w0 += 8;
                            }

                            T* d0 = out_at + i * output_w * 6 + j * 6;
                            T* d1 = d0 + 1;
                            T* d2 = d1 + 1;
                            T* d3 = d2 + 1;
                            T* d4 = d3 + 1;
                            T* d5 = d4 + 1;

                            for (int m = 0; m < 6; m++)
                            {
                                const T* tmp0 = tmp[m];

                                d0[0] = tmp0[0] + (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) + (tmp0[5] + tmp0[6]) * 32;
                                d1[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 2 + (tmp0[5] - tmp0[6]) * 16;
                                d2[0] = (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) * 4 + (tmp0[5] + tmp0[6]) * 8;
                                d3[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 8 + (tmp0[5] - tmp0[6]) * 4;
                                d4[0] = (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) * 16 + (tmp0[5] + tmp0[6]) * 2;
                                d5[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 32 + (tmp0[5] - tmp0[6]) + tmp0[7];

                                d0 += output_w;
                                d1 += output_w;
                                d2 += output_w;
                                d3 += output_w;
                                d4 += output_w;
                                d5 += output_w;

                            }

                        }
                    }
                }
            }

            inner_cut<T>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }

#ifdef TS_USE_SIMD
        template<>
        void Conv2dAlgorithm<float>::conv3x3_winograd63(const Tensor &x, const Tensor &k_tm, Tensor &out) {

            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 5) / 6 * 6;
            output_h = (output_h + 5) / 6 * 6;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 + 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 + 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<float>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);        

            //transform input data

            //const float BT[8][8] = {
            //    {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
            //
            //    {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
            //    {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
            //    {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
            //    {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
            //
            //    {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
            //};

            //const float32x4 f5(5.0f), f4(4.0f);
            //const float32x4 f2(2.0f), f2_5(2.5f);
            //const float32x4 f5_25(5.25f), f4_25(4.25f);
            //const float32x4 f1_25(1.25f), f0_5(0.5f);
            //const float32x4 f0_25(0.25f), f0(0.0f);
            //const float32x4 f8(8.0f), f16(16.0f), f32(32.0f);

            int w_tm = output_w / 6 * 8;
            int h_tm = output_h / 6 * 8;
            int col_blocks = w_tm / 8;
            int row_blocks = h_tm / 8;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 64 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 64 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const float* src_ptr = input_bordered.data<float>();
            float* dst_ptr = input_tm.data<float>();

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; c++)
                {
                    const float* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    float* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    float32x4 p1, p2, m1, m2, t1, t2;

                    //NOTE:I have no idea why i can't define these const variable out of the loop,
                    //if i do this,crash on android when i use openmp and neon,but it doesn't happen
                    //on x86(openmp+sse,avx)
                    const float32x4 f2(2.0f), f2_5(2.5f);
                    const float32x4 f5_25(5.25f), f4_25(4.25f);
                    const float32x4 f1_25(1.25f), f0_5(0.5f);
                    const float32x4 f4(4.0f), f0_25(0.25f);

                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const float* s0 = src_at + i * input_padded_w * 6 + j * 6;
                            const float* s1 = s0 + input_padded_w;
                            const float* s2 = s1 + input_padded_w;
                            const float* s3 = s2 + input_padded_w;
                            const float* s4 = s3 + input_padded_w;
                            const float* s5 = s4 + input_padded_w;
                            const float* s6 = s5 + input_padded_w;
                            const float* s7 = s6 + input_padded_w;

                            float32x4 l0(s0), r0(s0 + 4), l1(s1), r1(s1 + 4);
                            float32x4 l2(s2), r2(s2 + 4), l3(s3), r3(s3 + 4);
                            float32x4 l4(s4), r4(s4 + 4), l5(s5), r5(s5 + 4);
                            float32x4 l6(s6), r6(s6 + 4), l7(s7), r7(s7 + 4);

                            winograd_f63_input_transform(l0, l1, l2, l3, l4, l5, l6, l7,
                                t1, t2, m1, m2, p1, p2,
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);
                            transposex4x4(l0, l1, l2, l3);
                            transposex4x4(l4, l5, l6, l7);

                            winograd_f63_input_transform(r0, r1, r2, r3, r4, r5, r6, r7,
                                t1, t2, m1, m2, p1, p2,
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);
                            transposex4x4(r0, r1, r2, r3);
                            transposex4x4(r4, r5, r6, r7);

                            winograd_f63_input_transform(l0, l1, l2, l3, r0, r1, r2, r3,
                                t1, t2, m1, m2, p1, p2,
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);

                            winograd_f63_input_transform(l4, l5, l6, l7, r4, r5, r6, r7,
                                t1, t2, m1, m2, p1, p2,
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);

                            float* d0 = dst_at + (i * col_blocks + j) * 64;
                            float* d1 = d0 + 8;
                            float* d2 = d1 + 8;
                            float* d3 = d2 + 8;
                            float* d4 = d3 + 8;
                            float* d5 = d4 + 8;
                            float* d6 = d5 + 8;
                            float* d7 = d6 + 8;

                            l0.store(d0); l4.store(d0 + 4);
                            l1.store(d1); l5.store(d1 + 4);
                            l2.store(d2); l6.store(d2 + 4);
                            l3.store(d3); l7.store(d3 + 4);

                            r0.store(d4); r4.store(d4 + 4);
                            r1.store(d5); r5.store(d5 + 4);
                            r2.store(d6); r6.store(d6 + 4);
                            r3.store(d7); r7.store(d7 + 4);

                        }
                    }
                }
            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 64 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 64;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            float* out_tm_ptr = output_tm.data<float>();
            float* input_tm_ptr = dst_ptr;
            const float* kernel_tm_ptr = k_tm.data<float>();

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    float* out_tm_1 = out_tm_0 + outtm_c_offset;
                    float* out_tm_2 = out_tm_1 + outtm_c_offset;
                    float* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const float* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const float* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const float* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        float* out_0 = out_tm_0 + i * 64;
                        float* out_1 = out_tm_1 + i * 64;
                        float* out_2 = out_tm_2 + i * 64;
                        float* out_3 = out_tm_3 + i * 64;

                        float32x4x2 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f), sum04(0.f), sum05(0.f), sum06(0.f), sum07(0.f);
                        float32x4x2 sum10(0.f), sum11(0.f), sum12(0.f), sum13(0.f), sum14(0.f), sum15(0.f), sum16(0.f), sum17(0.f);
                        float32x4x2 sum20(0.f), sum21(0.f), sum22(0.f), sum23(0.f), sum24(0.f), sum25(0.f), sum26(0.f), sum27(0.f);
                        float32x4x2 sum30(0.f), sum31(0.f), sum32(0.f), sum33(0.f), sum34(0.f), sum35(0.f), sum36(0.f), sum37(0.f);

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;

                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 64;
                            const float* r1 = r0 + tm_c_offset;
                            const float* r2 = r1 + tm_c_offset;
                            const float* r3 = r2 + tm_c_offset;
                            float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);
                            float32x4x2 r10(r1), r11(r1 + 8), r12(r1 + 16), r13(r1 + 24), r14(r1 + 32), r15(r1 + 40), r16(r1 + 48), r17(r1 + 56);
                            float32x4x2 r20(r2), r21(r2 + 8), r22(r2 + 16), r23(r2 + 24), r24(r2 + 32), r25(r2 + 40), r26(r2 + 48), r27(r2 + 56);
                            float32x4x2 r30(r3), r31(r3 + 8), r32(r3 + 16), r33(r3 + 24), r34(r3 + 32), r35(r3 + 40), r36(r3 + 48), r37(r3 + 56);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const float* k3 = kernel_tm_3 + q * ktm_c_offset;

                            const float* k0c0 = k0, *k0c1 = k0c0 + 64, *k0c2 = k0c1 + 64, *k0c3 = k0c2 + 64;
                            const float* k1c0 = k1, *k1c1 = k1c0 + 64, *k1c2 = k1c1 + 64, *k1c3 = k1c2 + 64;
                            const float* k2c0 = k2, *k2c1 = k2c0 + 64, *k2c2 = k2c1 + 64, *k2c3 = k2c2 + 64;
                            const float* k3c0 = k3, *k3c1 = k3c0 + 64, *k3c2 = k3c1 + 64, *k3c3 = k3c2 + 64;

                            float32x4x2 k000(k0c0), k001(k0c0 + 8), k002(k0c0 + 16), k003(k0c0 + 24), k004(k0c0 + 32), k005(k0c0 + 40), k006(k0c0 + 48), k007(k0c0 + 56);
                            float32x4x2 k010(k0c1), k011(k0c1 + 8), k012(k0c1 + 16), k013(k0c1 + 24), k014(k0c1 + 32), k015(k0c1 + 40), k016(k0c1 + 48), k017(k0c1 + 56);
                            float32x4x2 k020(k0c2), k021(k0c2 + 8), k022(k0c2 + 16), k023(k0c2 + 24), k024(k0c2 + 32), k025(k0c2 + 40), k026(k0c2 + 48), k027(k0c2 + 56);
                            float32x4x2 k030(k0c3), k031(k0c3 + 8), k032(k0c3 + 16), k033(k0c3 + 24), k034(k0c3 + 32), k035(k0c3 + 40), k036(k0c3 + 48), k037(k0c3 + 56);

                            float32x4x2 k100(k1c0), k101(k1c0 + 8), k102(k1c0 + 16), k103(k1c0 + 24), k104(k1c0 + 32), k105(k1c0 + 40), k106(k1c0 + 48), k107(k1c0 + 56);
                            float32x4x2 k110(k1c1), k111(k1c1 + 8), k112(k1c1 + 16), k113(k1c1 + 24), k114(k1c1 + 32), k115(k1c1 + 40), k116(k1c1 + 48), k117(k1c1 + 56);
                            float32x4x2 k120(k1c2), k121(k1c2 + 8), k122(k1c2 + 16), k123(k1c2 + 24), k124(k1c2 + 32), k125(k1c2 + 40), k126(k1c2 + 48), k127(k1c2 + 56);
                            float32x4x2 k130(k1c3), k131(k1c3 + 8), k132(k1c3 + 16), k133(k1c3 + 24), k134(k1c3 + 32), k135(k1c3 + 40), k136(k1c3 + 48), k137(k1c3 + 56);

                            float32x4x2 k200(k2c0), k201(k2c0 + 8), k202(k2c0 + 16), k203(k2c0 + 24), k204(k2c0 + 32), k205(k2c0 + 40), k206(k2c0 + 48), k207(k2c0 + 56);
                            float32x4x2 k210(k2c1), k211(k2c1 + 8), k212(k2c1 + 16), k213(k2c1 + 24), k214(k2c1 + 32), k215(k2c1 + 40), k216(k2c1 + 48), k217(k2c1 + 56);
                            float32x4x2 k220(k2c2), k221(k2c2 + 8), k222(k2c2 + 16), k223(k2c2 + 24), k224(k2c2 + 32), k225(k2c2 + 40), k226(k2c2 + 48), k227(k2c2 + 56);
                            float32x4x2 k230(k2c3), k231(k2c3 + 8), k232(k2c3 + 16), k233(k2c3 + 24), k234(k2c3 + 32), k235(k2c3 + 40), k236(k2c3 + 48), k237(k2c3 + 56);

                            float32x4x2 k300(k3c0), k301(k3c0 + 8), k302(k3c0 + 16), k303(k3c0 + 24), k304(k3c0 + 32), k305(k3c0 + 40), k306(k3c0 + 48), k307(k3c0 + 56);
                            float32x4x2 k310(k3c1), k311(k3c1 + 8), k312(k3c1 + 16), k313(k3c1 + 24), k314(k3c1 + 32), k315(k3c1 + 40), k316(k3c1 + 48), k317(k3c1 + 56);
                            float32x4x2 k320(k3c2), k321(k3c2 + 8), k322(k3c2 + 16), k323(k3c2 + 24), k324(k3c2 + 32), k325(k3c2 + 40), k326(k3c2 + 48), k327(k3c2 + 56);
                            float32x4x2 k330(k3c3), k331(k3c3 + 8), k332(k3c3 + 16), k333(k3c3 + 24), k334(k3c3 + 32), k335(k3c3 + 40), k336(k3c3 + 48), k337(k3c3 + 56);

                            sum00 = fmadd(r00, k000, sum00); sum00 = fmadd(r10, k010, sum00); sum00 = fmadd(r20, k020, sum00); sum00 = fmadd(r30, k030, sum00);
                            sum01 = fmadd(r01, k001, sum01); sum01 = fmadd(r11, k011, sum01); sum01 = fmadd(r21, k021, sum01); sum01 = fmadd(r31, k031, sum01);
                            sum02 = fmadd(r02, k002, sum02); sum02 = fmadd(r12, k012, sum02); sum02 = fmadd(r22, k022, sum02); sum02 = fmadd(r32, k032, sum02);
                            sum03 = fmadd(r03, k003, sum03); sum03 = fmadd(r13, k013, sum03); sum03 = fmadd(r23, k023, sum03); sum03 = fmadd(r33, k033, sum03);
                            sum04 = fmadd(r04, k004, sum04); sum04 = fmadd(r14, k014, sum04); sum04 = fmadd(r24, k024, sum04); sum04 = fmadd(r34, k034, sum04);
                            sum05 = fmadd(r05, k005, sum05); sum05 = fmadd(r15, k015, sum05); sum05 = fmadd(r25, k025, sum05); sum05 = fmadd(r35, k035, sum05);
                            sum06 = fmadd(r06, k006, sum06); sum06 = fmadd(r16, k016, sum06); sum06 = fmadd(r26, k026, sum06); sum06 = fmadd(r36, k036, sum06);
                            sum07 = fmadd(r07, k007, sum07); sum07 = fmadd(r17, k017, sum07); sum07 = fmadd(r27, k027, sum07); sum07 = fmadd(r37, k037, sum07);

                            sum10 = fmadd(r00, k100, sum10); sum10 = fmadd(r10, k110, sum10); sum10 = fmadd(r20, k120, sum10); sum10 = fmadd(r30, k130, sum10);
                            sum11 = fmadd(r01, k101, sum11); sum11 = fmadd(r11, k111, sum11); sum11 = fmadd(r21, k121, sum11); sum11 = fmadd(r31, k131, sum11);
                            sum12 = fmadd(r02, k102, sum12); sum12 = fmadd(r12, k112, sum12); sum12 = fmadd(r22, k122, sum12); sum12 = fmadd(r32, k132, sum12);
                            sum13 = fmadd(r03, k103, sum13); sum13 = fmadd(r13, k113, sum13); sum13 = fmadd(r23, k123, sum13); sum13 = fmadd(r33, k133, sum13);
                            sum14 = fmadd(r04, k104, sum14); sum14 = fmadd(r14, k114, sum14); sum14 = fmadd(r24, k124, sum14); sum14 = fmadd(r34, k134, sum14);
                            sum15 = fmadd(r05, k105, sum15); sum15 = fmadd(r15, k115, sum15); sum15 = fmadd(r25, k125, sum15); sum15 = fmadd(r35, k135, sum15);
                            sum16 = fmadd(r06, k106, sum16); sum16 = fmadd(r16, k116, sum16); sum16 = fmadd(r26, k126, sum16); sum16 = fmadd(r36, k136, sum16);
                            sum17 = fmadd(r07, k107, sum17); sum17 = fmadd(r17, k117, sum17); sum17 = fmadd(r27, k127, sum17); sum17 = fmadd(r37, k137, sum17);

                            sum20 = fmadd(r00, k200, sum20); sum20 = fmadd(r10, k210, sum20); sum20 = fmadd(r20, k220, sum20); sum20 = fmadd(r30, k230, sum20);
                            sum21 = fmadd(r01, k201, sum21); sum21 = fmadd(r11, k211, sum21); sum21 = fmadd(r21, k221, sum21); sum21 = fmadd(r31, k231, sum21);
                            sum22 = fmadd(r02, k202, sum22); sum22 = fmadd(r12, k212, sum22); sum22 = fmadd(r22, k222, sum22); sum22 = fmadd(r32, k232, sum22);
                            sum23 = fmadd(r03, k203, sum23); sum23 = fmadd(r13, k213, sum23); sum23 = fmadd(r23, k223, sum23); sum23 = fmadd(r33, k233, sum23);
                            sum24 = fmadd(r04, k204, sum24); sum24 = fmadd(r14, k214, sum24); sum24 = fmadd(r24, k224, sum24); sum24 = fmadd(r34, k234, sum24);
                            sum25 = fmadd(r05, k205, sum25); sum25 = fmadd(r15, k215, sum25); sum25 = fmadd(r25, k225, sum25); sum25 = fmadd(r35, k235, sum25);
                            sum26 = fmadd(r06, k206, sum26); sum26 = fmadd(r16, k216, sum26); sum26 = fmadd(r26, k226, sum26); sum26 = fmadd(r36, k236, sum26);
                            sum27 = fmadd(r07, k207, sum27); sum27 = fmadd(r17, k217, sum27); sum27 = fmadd(r27, k227, sum27); sum27 = fmadd(r37, k237, sum27);

                            sum30 = fmadd(r00, k300, sum30); sum30 = fmadd(r10, k310, sum30); sum30 = fmadd(r20, k320, sum30); sum30 = fmadd(r30, k330, sum30);
                            sum31 = fmadd(r01, k301, sum31); sum31 = fmadd(r11, k311, sum31); sum31 = fmadd(r21, k321, sum31); sum31 = fmadd(r31, k331, sum31);
                            sum32 = fmadd(r02, k302, sum32); sum32 = fmadd(r12, k312, sum32); sum32 = fmadd(r22, k322, sum32); sum32 = fmadd(r32, k332, sum32);
                            sum33 = fmadd(r03, k303, sum33); sum33 = fmadd(r13, k313, sum33); sum33 = fmadd(r23, k323, sum33); sum33 = fmadd(r33, k333, sum33);
                            sum34 = fmadd(r04, k304, sum34); sum34 = fmadd(r14, k314, sum34); sum34 = fmadd(r24, k324, sum34); sum34 = fmadd(r34, k334, sum34);
                            sum35 = fmadd(r05, k305, sum35); sum35 = fmadd(r15, k315, sum35); sum35 = fmadd(r25, k325, sum35); sum35 = fmadd(r35, k335, sum35);
                            sum36 = fmadd(r06, k306, sum36); sum36 = fmadd(r16, k316, sum36); sum36 = fmadd(r26, k326, sum36); sum36 = fmadd(r36, k336, sum36);
                            sum37 = fmadd(r07, k307, sum37); sum37 = fmadd(r17, k317, sum37); sum37 = fmadd(r27, k327, sum37); sum37 = fmadd(r37, k337, sum37);


                        }

                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 64;
                            float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const float* k3 = kernel_tm_3 + q * ktm_c_offset;
                            float32x4x2 k000(k0), k001(k0 + 8), k002(k0 + 16), k003(k0 + 24), k004(k0 + 32), k005(k0 + 40), k006(k0 + 48), k007(k0 + 56);
                            float32x4x2 k100(k1), k101(k1 + 8), k102(k1 + 16), k103(k1 + 24), k104(k1 + 32), k105(k1 + 40), k106(k1 + 48), k107(k1 + 56);
                            float32x4x2 k200(k2), k201(k2 + 8), k202(k2 + 16), k203(k2 + 24), k204(k2 + 32), k205(k2 + 40), k206(k2 + 48), k207(k2 + 56);
                            float32x4x2 k300(k3), k301(k3 + 8), k302(k3 + 16), k303(k3 + 24), k304(k3 + 32), k305(k3 + 40), k306(k3 + 48), k307(k3 + 56);

                            sum00 = fmadd(r00, k000, sum00);
                            sum01 = fmadd(r01, k001, sum01);
                            sum02 = fmadd(r02, k002, sum02);
                            sum03 = fmadd(r03, k003, sum03);
                            sum04 = fmadd(r04, k004, sum04);
                            sum05 = fmadd(r05, k005, sum05);
                            sum06 = fmadd(r06, k006, sum06);
                            sum07 = fmadd(r07, k007, sum07);

                            sum10 = fmadd(r00, k100, sum10);
                            sum11 = fmadd(r01, k101, sum11);
                            sum12 = fmadd(r02, k102, sum12);
                            sum13 = fmadd(r03, k103, sum13);
                            sum14 = fmadd(r04, k104, sum14);
                            sum15 = fmadd(r05, k105, sum15);
                            sum16 = fmadd(r06, k106, sum16);
                            sum17 = fmadd(r07, k107, sum17);

                            sum20 = fmadd(r00, k200, sum20);
                            sum21 = fmadd(r01, k201, sum21);
                            sum22 = fmadd(r02, k202, sum22);
                            sum23 = fmadd(r03, k203, sum23);
                            sum24 = fmadd(r04, k204, sum24);
                            sum25 = fmadd(r05, k205, sum25);
                            sum26 = fmadd(r06, k206, sum26);
                            sum27 = fmadd(r07, k207, sum27);

                            sum30 = fmadd(r00, k300, sum30);
                            sum31 = fmadd(r01, k301, sum31);
                            sum32 = fmadd(r02, k302, sum32);
                            sum33 = fmadd(r03, k303, sum33);
                            sum34 = fmadd(r04, k304, sum34);
                            sum35 = fmadd(r05, k305, sum35);
                            sum36 = fmadd(r06, k306, sum36);
                            sum37 = fmadd(r07, k307, sum37);

                        }

                        sum00.store(out_0); sum01.store(out_0 + 8); sum02.store(out_0 + 16); sum03.store(out_0 + 24);
                        sum04.store(out_0 + 32); sum05.store(out_0 + 40); sum06.store(out_0 + 48); sum07.store(out_0 + 56);

                        sum10.store(out_1); sum11.store(out_1 + 8); sum12.store(out_1 + 16); sum13.store(out_1 + 24);
                        sum14.store(out_1 + 32); sum15.store(out_1 + 40); sum16.store(out_1 + 48); sum17.store(out_1 + 56);

                        sum20.store(out_2); sum21.store(out_2 + 8); sum22.store(out_2 + 16); sum23.store(out_2 + 24);
                        sum24.store(out_2 + 32); sum25.store(out_2 + 40); sum26.store(out_2 + 48); sum27.store(out_2 + 56);

                        sum30.store(out_3); sum31.store(out_3 + 8); sum32.store(out_3 + 16); sum33.store(out_3 + 24);
                        sum34.store(out_3 + 32); sum35.store(out_3 + 40); sum36.store(out_3 + 48); sum37.store(out_3 + 56);

                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = remain_outch; c < output_channel; c++)
                {
                    float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const float* kernel_tm_ptr = k_tm.data<float>();
                    const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        float* out_0 = out_tm_0 + i * 64;

                        float32x4x2 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f), sum04(0.f), sum05(0.f), sum06(0.f), sum07(0.f);

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 64;
                            const float* r1 = r0 + tm_c_offset;
                            const float* r2 = r1 + tm_c_offset;
                            const float* r3 = r2 + tm_c_offset;
                            float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);
                            float32x4x2 r10(r1), r11(r1 + 8), r12(r1 + 16), r13(r1 + 24), r14(r1 + 32), r15(r1 + 40), r16(r1 + 48), r17(r1 + 56);
                            float32x4x2 r20(r2), r21(r2 + 8), r22(r2 + 16), r23(r2 + 24), r24(r2 + 32), r25(r2 + 40), r26(r2 + 48), r27(r2 + 56);
                            float32x4x2 r30(r3), r31(r3 + 8), r32(r3 + 16), r33(r3 + 24), r34(r3 + 32), r35(r3 + 40), r36(r3 + 48), r37(r3 + 56);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = k0 + ktm_c_offset;
                            const float* k2 = k1 + ktm_c_offset;
                            const float* k3 = k2 + ktm_c_offset;

                            float32x4x2 k000(k0), k001(k0 + 8), k002(k0 + 16), k003(k0 + 24), k004(k0 + 32), k005(k0 + 40), k006(k0 + 48), k007(k0 + 56);
                            float32x4x2 k010(k1), k011(k1 + 8), k012(k1 + 16), k013(k1 + 24), k014(k1 + 32), k015(k1 + 40), k016(k1 + 48), k017(k1 + 56);
                            float32x4x2 k020(k2), k021(k2 + 8), k022(k2 + 16), k023(k2 + 24), k024(k2 + 32), k025(k2 + 40), k026(k2 + 48), k027(k2 + 56);
                            float32x4x2 k030(k3), k031(k3 + 8), k032(k3 + 16), k033(k3 + 24), k034(k3 + 32), k035(k3 + 40), k036(k3 + 48), k037(k3 + 56);

                            sum00 = fmadd(r00, k000, sum00); sum00 = fmadd(r10, k010, sum00); sum00 = fmadd(r20, k020, sum00); sum00 = fmadd(r30, k030, sum00);
                            sum01 = fmadd(r01, k001, sum01); sum01 = fmadd(r11, k011, sum01); sum01 = fmadd(r21, k021, sum01); sum01 = fmadd(r31, k031, sum01);
                            sum02 = fmadd(r02, k002, sum02); sum02 = fmadd(r12, k012, sum02); sum02 = fmadd(r22, k022, sum02); sum02 = fmadd(r32, k032, sum02);
                            sum03 = fmadd(r03, k003, sum03); sum03 = fmadd(r13, k013, sum03); sum03 = fmadd(r23, k023, sum03); sum03 = fmadd(r33, k033, sum03);
                            sum04 = fmadd(r04, k004, sum04); sum04 = fmadd(r14, k014, sum04); sum04 = fmadd(r24, k024, sum04); sum04 = fmadd(r34, k034, sum04);
                            sum05 = fmadd(r05, k005, sum05); sum05 = fmadd(r15, k015, sum05); sum05 = fmadd(r25, k025, sum05); sum05 = fmadd(r35, k035, sum05);
                            sum06 = fmadd(r06, k006, sum06); sum06 = fmadd(r16, k016, sum06); sum06 = fmadd(r26, k026, sum06); sum06 = fmadd(r36, k036, sum06);
                            sum07 = fmadd(r07, k007, sum07); sum07 = fmadd(r17, k017, sum07); sum07 = fmadd(r27, k027, sum07); sum07 = fmadd(r37, k037, sum07);
                        }

                        for (; q < input_channel; q++)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 64;
                            float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            float32x4x2 k000(k0), k001(k0 + 8), k002(k0 + 16), k003(k0 + 24), k004(k0 + 32), k005(k0 + 40), k006(k0 + 48), k007(k0 + 56);

                            sum00 = fmadd(r00, k000, sum00);
                            sum01 = fmadd(r01, k001, sum01);
                            sum02 = fmadd(r02, k002, sum02);
                            sum03 = fmadd(r03, k003, sum03);
                            sum04 = fmadd(r04, k004, sum04);
                            sum05 = fmadd(r05, k005, sum05);
                            sum06 = fmadd(r06, k006, sum06);
                            sum07 = fmadd(r07, k007, sum07);

                        }

                        sum00.store(out_0); sum01.store(out_0 + 8); sum02.store(out_0 + 16); sum03.store(out_0 + 24);
                        sum04.store(out_0 + 32); sum05.store(out_0 + 40); sum06.store(out_0 + 48); sum07.store(out_0 + 56);

                    }
                }
            }


            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            float* out_ptr = output_bordered.data<float>();

            //const float AT[6][8] = {
            //    {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
            //};

            // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
            // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
            // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
            // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
            // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
            // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

            //reuse (r1 + r2) (r1 - r2) (r3 + r4) (r3 - r4) (r5 + r6) (r5 - r6)

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < output_channel; c++)
                {
                    float* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    float* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    float32x4 tmp1add2, tmp1sub2;
                    float32x4 tmp3add4, tmp3sub4;
                    float32x4 tmp5add6, tmp5sub6;

                    const float32x4 f0(0.0f), f2(2.0f), f4(4.0f);
                    const float32x4 f8(8.0f), f16(16.0f), f32(32.0f);

                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const float* w0 = output_tm_at + (i * col_blocks + j) * 64;
                            const float* w1 = w0 + 8;
                            const float* w2 = w1 + 8;
                            const float* w3 = w2 + 8;
                            const float* w4 = w3 + 8;
                            const float* w5 = w4 + 8;
                            const float* w6 = w5 + 8;
                            const float* w7 = w6 + 8;

                            float32x4 l0(w0), r0(w0 + 4), l1(w1), r1(w1 + 4);
                            float32x4 l2(w2), r2(w2 + 4), l3(w3), r3(w3 + 4);
                            float32x4 l4(w4), r4(w4 + 4), l5(w5), r5(w5 + 4);
                            float32x4 l6(w6), r6(w6 + 4), l7(w7), r7(w7 + 4);

                            winograd_f63_output_transform(l0, l1, l2, l3, l4, l5, l6, l7,
                                tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                f0, f2, f4, f8, f16, f32);
                            transposex4x4(l0, l1, l2, l3);
                            transposex4x4(l4, l5, l6, l7);

                            winograd_f63_output_transform(r0, r1, r2, r3, r4, r5, r6, r7,
                                tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                f0, f2, f4, f8, f16, f32);
                            transposex4x4(r0, r1, r2, r3);
                            transposex4x4(r4, r5, r6, r7);

                            winograd_f63_output_transform(l0, l1, l2, l3, r0, r1, r2, r3,
                                tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                f0, f2, f4, f8, f16, f32);

                            winograd_f63_output_transform(l4, l5, l6, l7, r4, r5, r6, r7,
                                tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                f0, f2, f4, f8, f16, f32);

                            float* d0 = out_at + i * output_w * 6 + j * 6;
                            float* d1 = d0 + output_w;
                            float* d2 = d1 + output_w;
                            float* d3 = d2 + output_w;
                            float* d4 = d3 + output_w;
                            float* d5 = d4 + output_w;

                            if (((j * 6 + 6) > output_h) || ((i * 6 + 6) > output_w)) {
                                l0.store(d0); l4.store(d0 + 4);
                                l1.store(d1); l5.store(d1 + 4);
                                l2.store(d2); l6.store(d2 + 4);
                                l3.store(d3); l7.store(d3 + 4);

                                r0.store(d4); r4.store(d4 + 4);
                                r1.store(d5); r5.store(d5 + 4);
                            }
                            else {
                                l0.store(d0); *(d0 + 4) = *(((float*)&(l4.value))); *(d0 + 5) = *(((float*)&(l4.value)) + 1);
                                l1.store(d1); *(d1 + 4) = *(((float*)&(l5.value))); *(d1 + 5) = *(((float*)&(l5.value)) + 1);
                                l2.store(d2); *(d2 + 4) = *(((float*)&(l6.value))); *(d2 + 5) = *(((float*)&(l6.value)) + 1);
                                l3.store(d3); *(d3 + 4) = *(((float*)&(l7.value))); *(d3 + 5) = *(((float*)&(l7.value)) + 1);

                                r0.store(d4); *(d4 + 4) = *(((float*)&(r4.value))); *(d4 + 5) = *(((float*)&(r4.value)) + 1);
                                r1.store(d5); *(d5 + 4) = *(((float*)&(r5.value))); *(d5 + 5) = *(((float*)&(r5.value)) + 1);
                            }

                        }
                    }
                }
            }

            inner_cut<float>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }
#endif

        //TO DO: Support threadpool on template, only support float now
        template<typename T>
        void Conv2dAlgorithm<T>::conv3x3_winograd63_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out) {

            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 5) / 6 * 6;
            output_h = (output_h + 5) / 6 * 6;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<T>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            //const float BT[8][8] = {
            //    {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
            //
            //    {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
            //    {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
            //    {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
            //    {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
            //
            //    {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
            //};

            int w_tm = output_w / 6 * 8;
            int h_tm = output_h / 6 * 8;
            int col_blocks = w_tm / 8;
            int row_blocks = h_tm / 8;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 64 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 64 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const T* src_ptr = input_bordered.data<T>();
            T* dst_ptr = input_tm.data<T>();
            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < input_channel; c++)
                {
                    const T* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                    T* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                    T tmp[8][8];//save (d*B)T
                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const T* r0 = src_at + i * input_padded_w * 6 + j * 6;

                            for (int m = 0; m < 8; m++)
                            {
                                tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                                tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                                T tmp12_a = (r0[2] + r0[6] - r0[4] * 4.25f);
                                T tmp12_b = (r0[1] + r0[5] - r0[3] * 4.25f);

                                tmp[1][m] = tmp12_a + tmp12_b;
                                tmp[2][m] = tmp12_a - tmp12_b;

                                T tmp34_a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                                T tmp34_b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

                                tmp[3][m] = tmp34_a + tmp34_b;
                                tmp[4][m] = tmp34_a - tmp34_b;

                                T tmp56_a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                                T tmp56_b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                                tmp[5][m] = tmp56_a + tmp56_b;
                                tmp[6][m] = tmp56_a - tmp56_b;

                                r0 += input_padded_w;
                            }

                            T* d0 = dst_at + (i * col_blocks + j) * 64;

                            T* d1 = d0 + 1;
                            T* d2 = d1 + 1;
                            T* d3 = d2 + 1;
                            T* d4 = d3 + 1;
                            T* d5 = d4 + 1;
                            T* d6 = d5 + 1;
                            T* d7 = d6 + 1;

                            //(d*B)T * B == (BT*d*B)T == VT
                            for (int m = 0; m < 8; m++)
                            {
                                const T* tmp0 = tmp[m];

                                d0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * T(5.25);
                                d7[0] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * T(5.25);

                                T tmp12_a = (tmp0[2] + tmp0[6] - tmp0[4] * T(4.25));
                                T tmp12_b = (tmp0[1] - tmp0[3] * T(4.25) + tmp0[5]);

                                d1[0] = tmp12_a + tmp12_b;
                                d2[0] = tmp12_a - tmp12_b;

                                T tmp34_a = (tmp0[6] + tmp0[2] * T(0.25) - tmp0[4] * T(1.25));
                                T tmp34_b = (tmp0[1] * T(0.5) - tmp0[3] * T(2.5) + tmp0[5] * T(2));

                                d3[0] = tmp34_a + tmp34_b;
                                d4[0] = tmp34_a - tmp34_b;

                                T tmp56_a = (tmp0[6] + (tmp0[2] - tmp0[4] * T(1.25)) * T(4));
                                T tmp56_b = (tmp0[1] * T(2) - tmp0[3] * T(2.5) + tmp0[5] * T(0.5));

                                d5[0] = tmp56_a + tmp56_b;
                                d6[0] = tmp56_a - tmp56_b;

                                d0 += 8;
                                d1 += 8;
                                d2 += 8;
                                d3 += 8;
                                d4 += 8;
                                d5 += 8;
                                d6 += 8;
                                d7 += 8;

                            }
                        }
                    }
                }

            }

            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 64 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 64;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            T* out_tm_ptr = output_tm.data<T>();
            T* input_tm_ptr = dst_ptr;

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int cc = 0; cc < outch; cc++)
                {
                    int c = cc * 4;

                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_tm_1 = out_tm_0 + outtm_c_offset;
                    T* out_tm_2 = out_tm_1 + outtm_c_offset;
                    T* out_tm_3 = out_tm_2 + outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();

                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                    const T* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                    const T* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                    const T* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 64;
                        T* out_1 = out_tm_1 + i * 64;
                        T* out_2 = out_tm_2 + i * 64;
                        T* out_3 = out_tm_3 + i * 64;

                        T sum_0[64] = { T(0) };
                        T sum_1[64] = { T(0) };
                        T sum_2[64] = { T(0) };
                        T sum_3[64] = { T(0) };

                        int inputch = input_channel >> 2;
                        int remain_inputch = inputch << 2;

                        //#pragma omp parallel for num_threads(omp_get_max_threads())
                        for (int qq = 0; qq < inputch; qq++)
                        {
                            int q = qq * 4;
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;
                            const T* r1 = r0 + tm_c_offset;
                            const T* r2 = r1 + tm_c_offset;
                            const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r1[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r2[k] * k0[k];
                                k0 += 64;
                                sum_0[k] += r3[k] * k0[k];
                                k0 -= 192;

                                sum_1[k] += r0[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r1[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r2[k] * k1[k];
                                k1 += 64;
                                sum_1[k] += r3[k] * k1[k];
                                k1 -= 192;

                                sum_2[k] += r0[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r1[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r2[k] * k2[k];
                                k2 += 64;
                                sum_2[k] += r3[k] * k2[k];
                                k2 -= 192;

                                sum_3[k] += r0[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r1[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r2[k] * k3[k];
                                k3 += 64;
                                sum_3[k] += r3[k] * k3[k];
                                k3 -= 192;
                            }
                        }
                        for (int q = remain_inputch; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = kernel_tm_1 + q * ktm_c_offset;
                            const T* k2 = kernel_tm_2 + q * ktm_c_offset;
                            const T* k3 = kernel_tm_3 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_1[k] += r0[k] * k1[k];
                                sum_2[k] += r0[k] * k2[k];
                                sum_3[k] += r0[k] * k3[k];
                            }
                        }

                        for (int k = 0; k < 64; k++)
                        {
                            out_0[k] = sum_0[k];
                            out_1[k] = sum_1[k];
                            out_2[k] = sum_2[k];
                            out_3[k] = sum_3[k];
                        }

                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = remain_outch; c < output_channel; c++)
                {
                    T* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const T* kernel_tm_ptr = k_tm.data<T>();
                    const T* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        T* out_0 = out_tm_0 + i * 64;
                        T sum_0[64] = { T(0) };

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;
                            // const T* r1 = r0 + tm_c_offset;
                            // const T* r2 = r1 + tm_c_offset;
                            // const T* r3 = r2 + tm_c_offset;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const T* k1 = k0 + ktm_c_offset;
                            const T* k2 = k1 + ktm_c_offset;
                            const T* k3 = k2 + ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                                sum_0[k] += r0[k] * k1[k];
                                sum_0[k] += r0[k] * k2[k];
                                sum_0[k] += r0[k] * k3[k];
                            }
                        }

                        for (; q < input_channel; q++)
                        {
                            const T* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const T* r0 = input_tm_at + i * 64;

                            const T* k0 = kernel_tm_0 + q * ktm_c_offset;

                            for (int k = 0; k < 64; k++)
                            {
                                sum_0[k] += r0[k] * k0[k];
                            }
                        }

                        for (int k = 0; k < 64; k++)
                        {
                            out_0[k] = sum_0[k];
                        }

                    }
                }
            }

            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            T* out_ptr = output_bordered.data<T>();

            //const float AT[6][8] = {
            //    {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
            //};

            // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
            // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
            // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
            // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
            // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
            // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

            //reuse (r1 + r2) (r1 - r2) (r3 + r4) (r3 - r4) (r5 + r6) (r5 - r6)

            for (int n = 0; n < num; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < output_channel; c++)
                {
                    T* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                    T* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                    T tmp[6][8]; //(WT*A)T == AT*W
                    for (int i = 0; i < col_blocks; i++)
                    {
                        for (int j = 0; j < row_blocks; j++)
                        {
                            const T* w0 = output_tm_at + (i * col_blocks + j) * 64;

                            for (int m = 0; m < 8; m++)
                            {
                                T tmp1add2 = w0[1] + w0[2];
                                T tmp1sub2 = w0[1] - w0[2];
                                T tmp3add4 = w0[3] + w0[4];
                                T tmp3sub4 = w0[3] - w0[4];
                                T tmp5add6 = w0[5] + w0[6];
                                T tmp5sub6 = w0[5] - w0[6];

                                tmp[0][m] = w0[0] + tmp1add2 + tmp3add4 + tmp5add6 * 32;
                                tmp[1][m] = tmp1sub2 + tmp3sub4 * 2 + tmp5sub6 * 16;
                                tmp[2][m] = tmp1add2 + tmp3add4 * 4 + tmp5add6 * 8;
                                tmp[3][m] = tmp1sub2 + tmp3sub4 * 8 + tmp5sub6 * 4;
                                tmp[4][m] = tmp1add2 + tmp3add4 * 16 + tmp5add6 * 2;
                                tmp[5][m] = tmp1sub2 + tmp3sub4 * 32 + tmp5sub6 + w0[7];

                                //tmp[0][m] = w0[0] + (w0[1] + w0[2]) + (w0[3] + w0[4]) + (w0[5] + w0[6]) * 32;
                                //tmp[1][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 2 + (w0[5] - w0[6]) * 16;
                                //tmp[2][m] = (w0[1] + w0[2]) + (w0[3] + w0[4]) * 4 + (w0[5] + w0[6]) * 8;
                                //tmp[3][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 8 + (w0[5] - w0[6]) * 4;
                                //tmp[4][m] = (w0[1] + w0[2]) + (w0[3] + w0[4]) * 16 + (w0[5] + w0[6]) * 2;
                                //tmp[5][m] = (w0[1] - w0[2]) + (w0[3] - w0[4]) * 32 + (w0[5] - w0[6]) + w0[7];

                                w0 += 8;
                            }

                            T* d0 = out_at + i * output_w * 6 + j * 6;
                            T* d1 = d0 + 1;
                            T* d2 = d1 + 1;
                            T* d3 = d2 + 1;
                            T* d4 = d3 + 1;
                            T* d5 = d4 + 1;

                            for (int m = 0; m < 6; m++)
                            {
                                const T* tmp0 = tmp[m];

                                d0[0] = tmp0[0] + (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) + (tmp0[5] + tmp0[6]) * 32;
                                d1[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 2 + (tmp0[5] - tmp0[6]) * 16;
                                d2[0] = (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) * 4 + (tmp0[5] + tmp0[6]) * 8;
                                d3[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 8 + (tmp0[5] - tmp0[6]) * 4;
                                d4[0] = (tmp0[1] + tmp0[2]) + (tmp0[3] + tmp0[4]) * 16 + (tmp0[5] + tmp0[6]) * 2;
                                d5[0] = (tmp0[1] - tmp0[2]) + (tmp0[3] - tmp0[4]) * 32 + (tmp0[5] - tmp0[6]) + tmp0[7];

                                d0 += output_w;
                                d1 += output_w;
                                d2 += output_w;
                                d3 += output_w;
                                d4 += output_w;
                                d5 += output_w;

                            }

                        }
                    }
                }
            }

            inner_cut<T>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }

#ifdef TS_USE_SIMD
        template<>
        void Conv2dAlgorithm<float>::conv3x3_winograd63_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out) {

            auto input_shape = x.sizes();
            auto k_tm_shape = k_tm.sizes();
            auto out_shape = out.sizes();

            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channel = input_shape[1];
            int num = input_shape[0];

            int output_h = out_shape[2];
            int output_w = out_shape[3];
            int output_channel = out_shape[1];

            //pad
            output_w = (output_w + 5) / 6 * 6;
            output_h = (output_h + 5) / 6 * 6;

            int input_padded_w = output_w + 2;  //output_w = (input_w - 3)/1 - 1;
            int input_padded_h = output_h + 2;  //output_h = (input_h - 3)/1 - 1;

            Shape input_bordered_s = { num, input_channel, input_padded_h, input_padded_w };
            Tensor input_bordered(MemoryDevice(CPU), x.dtype(), input_bordered_s);
            int bordered_c_offset = input_padded_h * input_padded_w;
            int bordered_num_offset = input_channel * bordered_c_offset;

            inner_pad<float>(x, input_bordered, 0, input_padded_h - input_h, 0, input_padded_w - input_w, 0);

            //transform input data

            //const float BT[8][8] = {
            //    {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
            //
            //    {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
            //    {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
            //    {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
            //
            //    {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
            //    {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
            //
            //    {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
            //};

            //const float32x4 f5(5.0f), f4(4.0f);
            //const float32x4 f2(2.0f), f2_5(2.5f);
            //const float32x4 f5_25(5.25f), f4_25(4.25f);
            //const float32x4 f1_25(1.25f), f0_5(0.5f);
            //const float32x4 f0_25(0.25f), f0(0.0f);
            //const float32x4 f8(8.0f), f16(16.0f), f32(32.0f);

            int w_tm = output_w / 6 * 8;
            int h_tm = output_h / 6 * 8;
            int col_blocks = w_tm / 8;
            int row_blocks = h_tm / 8;
            int num_blocks = col_blocks * row_blocks;
            Shape input_tm_s = { num, input_channel, num_blocks, 64 };
            Tensor input_tm(MemoryDevice(CPU), x.dtype(), input_tm_s);
            int tm_c_offset = 64 * num_blocks;
            int tm_num_offset = input_channel * tm_c_offset;

            const float* src_ptr = input_bordered.data<float>();
            float* dst_ptr = input_tm.data<float>();

            auto thread_pool = ctx::lite::ptr<ThreadPool>();

            if (thread_pool == nullptr || thread_pool->size() <= 1)
            {
                for (int n = 0; n < num; n++)
                {
                    for (int c = 0; c < input_channel; c++)
                    {
                        const float* src_at = src_ptr + n * bordered_num_offset + c * bordered_c_offset;
                        float* dst_at = dst_ptr + n * tm_num_offset + c * tm_c_offset;

                        float32x4 p1, p2, m1, m2, t1, t2;
                        const float32x4 f5(5.0f), f4(4.0f);
                        const float32x4 f2(2.0f), f2_5(2.5f);
                        const float32x4 f5_25(5.25f), f4_25(4.25f);
                        const float32x4 f1_25(1.25f), f0_5(0.5f);
                        const float32x4 f0_25(0.25f), f0(0.0f);
                        const float32x4 f8(8.0f), f16(16.0f), f32(32.0f);

                        for (int i = 0; i < col_blocks; i++)
                        {
                            for (int j = 0; j < row_blocks; j++)
                            {
                                const float* s0 = src_at + i * input_padded_w * 6 + j * 6;
                                const float* s1 = s0 + input_padded_w;
                                const float* s2 = s1 + input_padded_w;
                                const float* s3 = s2 + input_padded_w;
                                const float* s4 = s3 + input_padded_w;
                                const float* s5 = s4 + input_padded_w;
                                const float* s6 = s5 + input_padded_w;
                                const float* s7 = s6 + input_padded_w;

                                float32x4 l0(s0), r0(s0 + 4), l1(s1), r1(s1 + 4);
                                float32x4 l2(s2), r2(s2 + 4), l3(s3), r3(s3 + 4);
                                float32x4 l4(s4), r4(s4 + 4), l5(s5), r5(s5 + 4);
                                float32x4 l6(s6), r6(s6 + 4), l7(s7), r7(s7 + 4);

                                winograd_f63_input_transform(l0, l1, l2, l3, l4, l5, l6, l7,
                                    t1, t2, m1, m2, p1, p2,
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);
                                transposex4x4(l0, l1, l2, l3);
                                transposex4x4(l4, l5, l6, l7);

                                winograd_f63_input_transform(r0, r1, r2, r3, r4, r5, r6, r7,
                                    t1, t2, m1, m2, p1, p2,
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);
                                transposex4x4(r0, r1, r2, r3);
                                transposex4x4(r4, r5, r6, r7);

                                winograd_f63_input_transform(l0, l1, l2, l3, r0, r1, r2, r3,
                                    t1, t2, m1, m2, p1, p2,
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);

                                winograd_f63_input_transform(l4, l5, l6, l7, r4, r5, r6, r7,
                                    t1, t2, m1, m2, p1, p2,
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);

                                float* d0 = dst_at + (i * col_blocks + j) * 64;
                                float* d1 = d0 + 8;
                                float* d2 = d1 + 8;
                                float* d3 = d2 + 8;
                                float* d4 = d3 + 8;
                                float* d5 = d4 + 8;
                                float* d6 = d5 + 8;
                                float* d7 = d6 + 8;

                                l0.store(d0); l4.store(d0 + 4);
                                l1.store(d1); l5.store(d1 + 4);
                                l2.store(d2); l6.store(d2 + 4);
                                l3.store(d3); l7.store(d3 + 4);

                                r0.store(d4); r4.store(d4 + 4);
                                r1.store(d5); r5.store(d5 + 4);
                                r2.store(d6); r6.store(d6 + 4);
                                r3.store(d7); r7.store(d7 + 4);

                            }
                        }
                    }
                }
            }
            else
            {
                for (int n = 0; n < num; n++)
                {
                    auto bins = split_bins(0, input_channel, int(thread_pool->size()));
                    for (auto &bin : bins)
                    {
                        thread_pool->run([&, n, src_ptr, dst_ptr, bin](int) {
                            const float* src_at = src_ptr + n * bordered_num_offset + bin.first * bordered_c_offset;
                            float* dst_at = dst_ptr + n * tm_num_offset + bin.first * tm_c_offset;
                            for (int c = bin.first; c < bin.second; c++)
                            {

                                float32x4 p1, p2, m1, m2, t1, t2;
                                const float32x4 f5(5.0f), f4(4.0f);
                                const float32x4 f2(2.0f), f2_5(2.5f);
                                const float32x4 f5_25(5.25f), f4_25(4.25f);
                                const float32x4 f1_25(1.25f), f0_5(0.5f);
                                const float32x4 f0_25(0.25f), f0(0.0f);
                                const float32x4 f8(8.0f), f16(16.0f), f32(32.0f);

                                for (int i = 0; i < col_blocks; i++)
                                {
                                    for (int j = 0; j < row_blocks; j++)
                                    {
                                        const float* s0 = src_at + i * input_padded_w * 6 + j * 6;
                                        const float* s1 = s0 + input_padded_w;
                                        const float* s2 = s1 + input_padded_w;
                                        const float* s3 = s2 + input_padded_w;
                                        const float* s4 = s3 + input_padded_w;
                                        const float* s5 = s4 + input_padded_w;
                                        const float* s6 = s5 + input_padded_w;
                                        const float* s7 = s6 + input_padded_w;

                                        float32x4 l0(s0), r0(s0 + 4), l1(s1), r1(s1 + 4);
                                        float32x4 l2(s2), r2(s2 + 4), l3(s3), r3(s3 + 4);
                                        float32x4 l4(s4), r4(s4 + 4), l5(s5), r5(s5 + 4);
                                        float32x4 l6(s6), r6(s6 + 4), l7(s7), r7(s7 + 4);

                                        winograd_f63_input_transform(l0, l1, l2, l3, l4, l5, l6, l7,
                                            t1, t2, m1, m2, p1, p2,
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);
                                        transposex4x4(l0, l1, l2, l3);
                                        transposex4x4(l4, l5, l6, l7);

                                        winograd_f63_input_transform(r0, r1, r2, r3, r4, r5, r6, r7,
                                            t1, t2, m1, m2, p1, p2,
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);
                                        transposex4x4(r0, r1, r2, r3);
                                        transposex4x4(r4, r5, r6, r7);

                                        winograd_f63_input_transform(l0, l1, l2, l3, r0, r1, r2, r3,
                                            t1, t2, m1, m2, p1, p2,
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);

                                        winograd_f63_input_transform(l4, l5, l6, l7, r4, r5, r6, r7,
                                            t1, t2, m1, m2, p1, p2,
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25);

                                        float* d0 = dst_at + (i * col_blocks + j) * 64;
                                        float* d1 = d0 + 8;
                                        float* d2 = d1 + 8;
                                        float* d3 = d2 + 8;
                                        float* d4 = d3 + 8;
                                        float* d5 = d4 + 8;
                                        float* d6 = d5 + 8;
                                        float* d7 = d6 + 8;

                                        l0.store(d0); l4.store(d0 + 4);
                                        l1.store(d1); l5.store(d1 + 4);
                                        l2.store(d2); l6.store(d2 + 4);
                                        l3.store(d3); l7.store(d3 + 4);

                                        r0.store(d4); r4.store(d4 + 4);
                                        r1.store(d5); r5.store(d5 + 4);
                                        r2.store(d6); r6.store(d6 + 4);
                                        r3.store(d7); r7.store(d7 + 4);

                                    }
                                }
                                src_at += bordered_c_offset;
                                dst_at += tm_c_offset;
                            }
                        });
                    }

                }
                thread_pool->join();
            }


            //begin dot
            Shape out_tm_s = { num, output_channel, num_blocks, 64 };
            Tensor output_tm(MemoryDevice(CPU), x.dtype(), out_tm_s);
            int outtm_c_offset = num_blocks * 64;
            int outtm_n_offset = output_channel * outtm_c_offset;

            int ktm_c_offset = k_tm_shape[2] * k_tm_shape[3];
            int ktm_n_offset = k_tm_shape[1] * ktm_c_offset;

            int outch = output_channel >> 2;
            int remain_outch = outch << 2;

            float* out_tm_ptr = output_tm.data<float>();
            float* input_tm_ptr = dst_ptr;
            const float* kernel_tm_ptr = k_tm.data<float>();

            for (int n = 0; n < num; n++)
            {
                if (thread_pool == nullptr || thread_pool->size() <= 1)
                {
                    for (int cc = 0; cc < outch; cc++)
                    {
                        int c = cc * 4;

                        float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                        float* out_tm_1 = out_tm_0 + outtm_c_offset;
                        float* out_tm_2 = out_tm_1 + outtm_c_offset;
                        float* out_tm_3 = out_tm_2 + outtm_c_offset;

                        const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;
                        const float* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                        const float* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                        const float* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                        for (int i = 0; i < num_blocks; i++)
                        {
                            float* out_0 = out_tm_0 + i * 64;
                            float* out_1 = out_tm_1 + i * 64;
                            float* out_2 = out_tm_2 + i * 64;
                            float* out_3 = out_tm_3 + i * 64;

                            float32x4x2 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f), sum04(0.f), sum05(0.f), sum06(0.f), sum07(0.f);
                            float32x4x2 sum10(0.f), sum11(0.f), sum12(0.f), sum13(0.f), sum14(0.f), sum15(0.f), sum16(0.f), sum17(0.f);
                            float32x4x2 sum20(0.f), sum21(0.f), sum22(0.f), sum23(0.f), sum24(0.f), sum25(0.f), sum26(0.f), sum27(0.f);
                            float32x4x2 sum30(0.f), sum31(0.f), sum32(0.f), sum33(0.f), sum34(0.f), sum35(0.f), sum36(0.f), sum37(0.f);

                            int inputch = input_channel >> 2;
                            int remain_inputch = inputch << 2;

                            //#pragma omp parallel for num_threads(omp_get_max_threads())
                            for (int qq = 0; qq < inputch; qq++)
                            {
                                int q = qq * 4;
                                const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                const float* r0 = input_tm_at + i * 64;
                                const float* r1 = r0 + tm_c_offset;
                                const float* r2 = r1 + tm_c_offset;
                                const float* r3 = r2 + tm_c_offset;
                                float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);
                                float32x4x2 r10(r1), r11(r1 + 8), r12(r1 + 16), r13(r1 + 24), r14(r1 + 32), r15(r1 + 40), r16(r1 + 48), r17(r1 + 56);
                                float32x4x2 r20(r2), r21(r2 + 8), r22(r2 + 16), r23(r2 + 24), r24(r2 + 32), r25(r2 + 40), r26(r2 + 48), r27(r2 + 56);
                                float32x4x2 r30(r3), r31(r3 + 8), r32(r3 + 16), r33(r3 + 24), r34(r3 + 32), r35(r3 + 40), r36(r3 + 48), r37(r3 + 56);

                                const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                const float* k3 = kernel_tm_3 + q * ktm_c_offset;

                                const float* k0c0 = k0, *k0c1 = k0c0 + 64, *k0c2 = k0c1 + 64, *k0c3 = k0c2 + 64;
                                const float* k1c0 = k1, *k1c1 = k1c0 + 64, *k1c2 = k1c1 + 64, *k1c3 = k1c2 + 64;
                                const float* k2c0 = k2, *k2c1 = k2c0 + 64, *k2c2 = k2c1 + 64, *k2c3 = k2c2 + 64;
                                const float* k3c0 = k3, *k3c1 = k3c0 + 64, *k3c2 = k3c1 + 64, *k3c3 = k3c2 + 64;

                                float32x4x2 k000(k0c0), k001(k0c0 + 8), k002(k0c0 + 16), k003(k0c0 + 24), k004(k0c0 + 32), k005(k0c0 + 40), k006(k0c0 + 48), k007(k0c0 + 56);
                                float32x4x2 k010(k0c1), k011(k0c1 + 8), k012(k0c1 + 16), k013(k0c1 + 24), k014(k0c1 + 32), k015(k0c1 + 40), k016(k0c1 + 48), k017(k0c1 + 56);
                                float32x4x2 k020(k0c2), k021(k0c2 + 8), k022(k0c2 + 16), k023(k0c2 + 24), k024(k0c2 + 32), k025(k0c2 + 40), k026(k0c2 + 48), k027(k0c2 + 56);
                                float32x4x2 k030(k0c3), k031(k0c3 + 8), k032(k0c3 + 16), k033(k0c3 + 24), k034(k0c3 + 32), k035(k0c3 + 40), k036(k0c3 + 48), k037(k0c3 + 56);

                                float32x4x2 k100(k1c0), k101(k1c0 + 8), k102(k1c0 + 16), k103(k1c0 + 24), k104(k1c0 + 32), k105(k1c0 + 40), k106(k1c0 + 48), k107(k1c0 + 56);
                                float32x4x2 k110(k1c1), k111(k1c1 + 8), k112(k1c1 + 16), k113(k1c1 + 24), k114(k1c1 + 32), k115(k1c1 + 40), k116(k1c1 + 48), k117(k1c1 + 56);
                                float32x4x2 k120(k1c2), k121(k1c2 + 8), k122(k1c2 + 16), k123(k1c2 + 24), k124(k1c2 + 32), k125(k1c2 + 40), k126(k1c2 + 48), k127(k1c2 + 56);
                                float32x4x2 k130(k1c3), k131(k1c3 + 8), k132(k1c3 + 16), k133(k1c3 + 24), k134(k1c3 + 32), k135(k1c3 + 40), k136(k1c3 + 48), k137(k1c3 + 56);

                                float32x4x2 k200(k2c0), k201(k2c0 + 8), k202(k2c0 + 16), k203(k2c0 + 24), k204(k2c0 + 32), k205(k2c0 + 40), k206(k2c0 + 48), k207(k2c0 + 56);
                                float32x4x2 k210(k2c1), k211(k2c1 + 8), k212(k2c1 + 16), k213(k2c1 + 24), k214(k2c1 + 32), k215(k2c1 + 40), k216(k2c1 + 48), k217(k2c1 + 56);
                                float32x4x2 k220(k2c2), k221(k2c2 + 8), k222(k2c2 + 16), k223(k2c2 + 24), k224(k2c2 + 32), k225(k2c2 + 40), k226(k2c2 + 48), k227(k2c2 + 56);
                                float32x4x2 k230(k2c3), k231(k2c3 + 8), k232(k2c3 + 16), k233(k2c3 + 24), k234(k2c3 + 32), k235(k2c3 + 40), k236(k2c3 + 48), k237(k2c3 + 56);

                                float32x4x2 k300(k3c0), k301(k3c0 + 8), k302(k3c0 + 16), k303(k3c0 + 24), k304(k3c0 + 32), k305(k3c0 + 40), k306(k3c0 + 48), k307(k3c0 + 56);
                                float32x4x2 k310(k3c1), k311(k3c1 + 8), k312(k3c1 + 16), k313(k3c1 + 24), k314(k3c1 + 32), k315(k3c1 + 40), k316(k3c1 + 48), k317(k3c1 + 56);
                                float32x4x2 k320(k3c2), k321(k3c2 + 8), k322(k3c2 + 16), k323(k3c2 + 24), k324(k3c2 + 32), k325(k3c2 + 40), k326(k3c2 + 48), k327(k3c2 + 56);
                                float32x4x2 k330(k3c3), k331(k3c3 + 8), k332(k3c3 + 16), k333(k3c3 + 24), k334(k3c3 + 32), k335(k3c3 + 40), k336(k3c3 + 48), k337(k3c3 + 56);

                                sum00 = fmadd(r00, k000, sum00); sum00 = fmadd(r10, k010, sum00); sum00 = fmadd(r20, k020, sum00); sum00 = fmadd(r30, k030, sum00);
                                sum01 = fmadd(r01, k001, sum01); sum01 = fmadd(r11, k011, sum01); sum01 = fmadd(r21, k021, sum01); sum01 = fmadd(r31, k031, sum01);
                                sum02 = fmadd(r02, k002, sum02); sum02 = fmadd(r12, k012, sum02); sum02 = fmadd(r22, k022, sum02); sum02 = fmadd(r32, k032, sum02);
                                sum03 = fmadd(r03, k003, sum03); sum03 = fmadd(r13, k013, sum03); sum03 = fmadd(r23, k023, sum03); sum03 = fmadd(r33, k033, sum03);
                                sum04 = fmadd(r04, k004, sum04); sum04 = fmadd(r14, k014, sum04); sum04 = fmadd(r24, k024, sum04); sum04 = fmadd(r34, k034, sum04);
                                sum05 = fmadd(r05, k005, sum05); sum05 = fmadd(r15, k015, sum05); sum05 = fmadd(r25, k025, sum05); sum05 = fmadd(r35, k035, sum05);
                                sum06 = fmadd(r06, k006, sum06); sum06 = fmadd(r16, k016, sum06); sum06 = fmadd(r26, k026, sum06); sum06 = fmadd(r36, k036, sum06);
                                sum07 = fmadd(r07, k007, sum07); sum07 = fmadd(r17, k017, sum07); sum07 = fmadd(r27, k027, sum07); sum07 = fmadd(r37, k037, sum07);

                                sum10 = fmadd(r00, k100, sum10); sum10 = fmadd(r10, k110, sum10); sum10 = fmadd(r20, k120, sum10); sum10 = fmadd(r30, k130, sum10);
                                sum11 = fmadd(r01, k101, sum11); sum11 = fmadd(r11, k111, sum11); sum11 = fmadd(r21, k121, sum11); sum11 = fmadd(r31, k131, sum11);
                                sum12 = fmadd(r02, k102, sum12); sum12 = fmadd(r12, k112, sum12); sum12 = fmadd(r22, k122, sum12); sum12 = fmadd(r32, k132, sum12);
                                sum13 = fmadd(r03, k103, sum13); sum13 = fmadd(r13, k113, sum13); sum13 = fmadd(r23, k123, sum13); sum13 = fmadd(r33, k133, sum13);
                                sum14 = fmadd(r04, k104, sum14); sum14 = fmadd(r14, k114, sum14); sum14 = fmadd(r24, k124, sum14); sum14 = fmadd(r34, k134, sum14);
                                sum15 = fmadd(r05, k105, sum15); sum15 = fmadd(r15, k115, sum15); sum15 = fmadd(r25, k125, sum15); sum15 = fmadd(r35, k135, sum15);
                                sum16 = fmadd(r06, k106, sum16); sum16 = fmadd(r16, k116, sum16); sum16 = fmadd(r26, k126, sum16); sum16 = fmadd(r36, k136, sum16);
                                sum17 = fmadd(r07, k107, sum17); sum17 = fmadd(r17, k117, sum17); sum17 = fmadd(r27, k127, sum17); sum17 = fmadd(r37, k137, sum17);

                                sum20 = fmadd(r00, k200, sum20); sum20 = fmadd(r10, k210, sum20); sum20 = fmadd(r20, k220, sum20); sum20 = fmadd(r30, k230, sum20);
                                sum21 = fmadd(r01, k201, sum21); sum21 = fmadd(r11, k211, sum21); sum21 = fmadd(r21, k221, sum21); sum21 = fmadd(r31, k231, sum21);
                                sum22 = fmadd(r02, k202, sum22); sum22 = fmadd(r12, k212, sum22); sum22 = fmadd(r22, k222, sum22); sum22 = fmadd(r32, k232, sum22);
                                sum23 = fmadd(r03, k203, sum23); sum23 = fmadd(r13, k213, sum23); sum23 = fmadd(r23, k223, sum23); sum23 = fmadd(r33, k233, sum23);
                                sum24 = fmadd(r04, k204, sum24); sum24 = fmadd(r14, k214, sum24); sum24 = fmadd(r24, k224, sum24); sum24 = fmadd(r34, k234, sum24);
                                sum25 = fmadd(r05, k205, sum25); sum25 = fmadd(r15, k215, sum25); sum25 = fmadd(r25, k225, sum25); sum25 = fmadd(r35, k235, sum25);
                                sum26 = fmadd(r06, k206, sum26); sum26 = fmadd(r16, k216, sum26); sum26 = fmadd(r26, k226, sum26); sum26 = fmadd(r36, k236, sum26);
                                sum27 = fmadd(r07, k207, sum27); sum27 = fmadd(r17, k217, sum27); sum27 = fmadd(r27, k227, sum27); sum27 = fmadd(r37, k237, sum27);

                                sum30 = fmadd(r00, k300, sum30); sum30 = fmadd(r10, k310, sum30); sum30 = fmadd(r20, k320, sum30); sum30 = fmadd(r30, k330, sum30);
                                sum31 = fmadd(r01, k301, sum31); sum31 = fmadd(r11, k311, sum31); sum31 = fmadd(r21, k321, sum31); sum31 = fmadd(r31, k331, sum31);
                                sum32 = fmadd(r02, k302, sum32); sum32 = fmadd(r12, k312, sum32); sum32 = fmadd(r22, k322, sum32); sum32 = fmadd(r32, k332, sum32);
                                sum33 = fmadd(r03, k303, sum33); sum33 = fmadd(r13, k313, sum33); sum33 = fmadd(r23, k323, sum33); sum33 = fmadd(r33, k333, sum33);
                                sum34 = fmadd(r04, k304, sum34); sum34 = fmadd(r14, k314, sum34); sum34 = fmadd(r24, k324, sum34); sum34 = fmadd(r34, k334, sum34);
                                sum35 = fmadd(r05, k305, sum35); sum35 = fmadd(r15, k315, sum35); sum35 = fmadd(r25, k325, sum35); sum35 = fmadd(r35, k335, sum35);
                                sum36 = fmadd(r06, k306, sum36); sum36 = fmadd(r16, k316, sum36); sum36 = fmadd(r26, k326, sum36); sum36 = fmadd(r36, k336, sum36);
                                sum37 = fmadd(r07, k307, sum37); sum37 = fmadd(r17, k317, sum37); sum37 = fmadd(r27, k327, sum37); sum37 = fmadd(r37, k337, sum37);


                            }

                            for (int q = remain_inputch; q < input_channel; q++)
                            {
                                const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                const float* r0 = input_tm_at + i * 64;
                                float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);

                                const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                const float* k3 = kernel_tm_3 + q * ktm_c_offset;
                                float32x4x2 k000(k0), k001(k0 + 8), k002(k0 + 16), k003(k0 + 24), k004(k0 + 32), k005(k0 + 40), k006(k0 + 48), k007(k0 + 56);
                                float32x4x2 k100(k1), k101(k1 + 8), k102(k1 + 16), k103(k1 + 24), k104(k1 + 32), k105(k1 + 40), k106(k1 + 48), k107(k1 + 56);
                                float32x4x2 k200(k2), k201(k2 + 8), k202(k2 + 16), k203(k2 + 24), k204(k2 + 32), k205(k2 + 40), k206(k2 + 48), k207(k2 + 56);
                                float32x4x2 k300(k3), k301(k3 + 8), k302(k3 + 16), k303(k3 + 24), k304(k3 + 32), k305(k3 + 40), k306(k3 + 48), k307(k3 + 56);

                                sum00 = fmadd(r00, k000, sum00);
                                sum01 = fmadd(r01, k001, sum01);
                                sum02 = fmadd(r02, k002, sum02);
                                sum03 = fmadd(r03, k003, sum03);
                                sum04 = fmadd(r04, k004, sum04);
                                sum05 = fmadd(r05, k005, sum05);
                                sum06 = fmadd(r06, k006, sum06);
                                sum07 = fmadd(r07, k007, sum07);

                                sum10 = fmadd(r00, k100, sum10);
                                sum11 = fmadd(r01, k101, sum11);
                                sum12 = fmadd(r02, k102, sum12);
                                sum13 = fmadd(r03, k103, sum13);
                                sum14 = fmadd(r04, k104, sum14);
                                sum15 = fmadd(r05, k105, sum15);
                                sum16 = fmadd(r06, k106, sum16);
                                sum17 = fmadd(r07, k107, sum17);

                                sum20 = fmadd(r00, k200, sum20);
                                sum21 = fmadd(r01, k201, sum21);
                                sum22 = fmadd(r02, k202, sum22);
                                sum23 = fmadd(r03, k203, sum23);
                                sum24 = fmadd(r04, k204, sum24);
                                sum25 = fmadd(r05, k205, sum25);
                                sum26 = fmadd(r06, k206, sum26);
                                sum27 = fmadd(r07, k207, sum27);

                                sum30 = fmadd(r00, k300, sum30);
                                sum31 = fmadd(r01, k301, sum31);
                                sum32 = fmadd(r02, k302, sum32);
                                sum33 = fmadd(r03, k303, sum33);
                                sum34 = fmadd(r04, k304, sum34);
                                sum35 = fmadd(r05, k305, sum35);
                                sum36 = fmadd(r06, k306, sum36);
                                sum37 = fmadd(r07, k307, sum37);

                            }

                            sum00.store(out_0); sum01.store(out_0 + 8); sum02.store(out_0 + 16); sum03.store(out_0 + 24);
                            sum04.store(out_0 + 32); sum05.store(out_0 + 40); sum06.store(out_0 + 48); sum07.store(out_0 + 56);

                            sum10.store(out_1); sum11.store(out_1 + 8); sum12.store(out_1 + 16); sum13.store(out_1 + 24);
                            sum14.store(out_1 + 32); sum15.store(out_1 + 40); sum16.store(out_1 + 48); sum17.store(out_1 + 56);

                            sum20.store(out_2); sum21.store(out_2 + 8); sum22.store(out_2 + 16); sum23.store(out_2 + 24);
                            sum24.store(out_2 + 32); sum25.store(out_2 + 40); sum26.store(out_2 + 48); sum27.store(out_2 + 56);

                            sum30.store(out_3); sum31.store(out_3 + 8); sum32.store(out_3 + 16); sum33.store(out_3 + 24);
                            sum34.store(out_3 + 32); sum35.store(out_3 + 40); sum36.store(out_3 + 48); sum37.store(out_3 + 56);

                        }
                    }
                }
                else
                {
                    auto bins = split_bins(0, outch, (int)thread_pool->size());
                    for (auto &bin : bins)
                    {
                        thread_pool->run([&, n, input_tm_ptr, out_tm_ptr, kernel_tm_ptr, bin](int) {

                            float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + 4 * bin.first * outtm_c_offset;

                            const float* kernel_tm_0 = kernel_tm_ptr + 4 * bin.first * ktm_n_offset;

                            for (int c = bin.first; c < bin.second; c++)
                            {
                                float* out_tm_1 = out_tm_0 + outtm_c_offset;
                                float* out_tm_2 = out_tm_1 + outtm_c_offset;
                                float* out_tm_3 = out_tm_2 + outtm_c_offset;

                                const float* kernel_tm_1 = kernel_tm_0 + ktm_n_offset;
                                const float* kernel_tm_2 = kernel_tm_1 + ktm_n_offset;
                                const float* kernel_tm_3 = kernel_tm_2 + ktm_n_offset;

                                for (int i = 0; i < num_blocks; i++)
                                {
                                    float* out_0 = out_tm_0 + i * 64;
                                    float* out_1 = out_tm_1 + i * 64;
                                    float* out_2 = out_tm_2 + i * 64;
                                    float* out_3 = out_tm_3 + i * 64;

                                    float32x4x2 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f), sum04(0.f), sum05(0.f), sum06(0.f), sum07(0.f);
                                    float32x4x2 sum10(0.f), sum11(0.f), sum12(0.f), sum13(0.f), sum14(0.f), sum15(0.f), sum16(0.f), sum17(0.f);
                                    float32x4x2 sum20(0.f), sum21(0.f), sum22(0.f), sum23(0.f), sum24(0.f), sum25(0.f), sum26(0.f), sum27(0.f);
                                    float32x4x2 sum30(0.f), sum31(0.f), sum32(0.f), sum33(0.f), sum34(0.f), sum35(0.f), sum36(0.f), sum37(0.f);

                                    int inputch = input_channel >> 2;
                                    int remain_inputch = inputch << 2;

                                    //#pragma omp parallel for num_threads(omp_get_max_threads())
                                    for (int qq = 0; qq < inputch; qq++)
                                    {
                                        int q = qq * 4;
                                        const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                        const float* r0 = input_tm_at + i * 64;
                                        const float* r1 = r0 + tm_c_offset;
                                        const float* r2 = r1 + tm_c_offset;
                                        const float* r3 = r2 + tm_c_offset;
                                        float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);
                                        float32x4x2 r10(r1), r11(r1 + 8), r12(r1 + 16), r13(r1 + 24), r14(r1 + 32), r15(r1 + 40), r16(r1 + 48), r17(r1 + 56);
                                        float32x4x2 r20(r2), r21(r2 + 8), r22(r2 + 16), r23(r2 + 24), r24(r2 + 32), r25(r2 + 40), r26(r2 + 48), r27(r2 + 56);
                                        float32x4x2 r30(r3), r31(r3 + 8), r32(r3 + 16), r33(r3 + 24), r34(r3 + 32), r35(r3 + 40), r36(r3 + 48), r37(r3 + 56);

                                        const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                        const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                        const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                        const float* k3 = kernel_tm_3 + q * ktm_c_offset;

                                        const float* k0c0 = k0, *k0c1 = k0c0 + 64, *k0c2 = k0c1 + 64, *k0c3 = k0c2 + 64;
                                        const float* k1c0 = k1, *k1c1 = k1c0 + 64, *k1c2 = k1c1 + 64, *k1c3 = k1c2 + 64;
                                        const float* k2c0 = k2, *k2c1 = k2c0 + 64, *k2c2 = k2c1 + 64, *k2c3 = k2c2 + 64;
                                        const float* k3c0 = k3, *k3c1 = k3c0 + 64, *k3c2 = k3c1 + 64, *k3c3 = k3c2 + 64;

                                        float32x4x2 k000(k0c0), k001(k0c0 + 8), k002(k0c0 + 16), k003(k0c0 + 24), k004(k0c0 + 32), k005(k0c0 + 40), k006(k0c0 + 48), k007(k0c0 + 56);
                                        float32x4x2 k010(k0c1), k011(k0c1 + 8), k012(k0c1 + 16), k013(k0c1 + 24), k014(k0c1 + 32), k015(k0c1 + 40), k016(k0c1 + 48), k017(k0c1 + 56);
                                        float32x4x2 k020(k0c2), k021(k0c2 + 8), k022(k0c2 + 16), k023(k0c2 + 24), k024(k0c2 + 32), k025(k0c2 + 40), k026(k0c2 + 48), k027(k0c2 + 56);
                                        float32x4x2 k030(k0c3), k031(k0c3 + 8), k032(k0c3 + 16), k033(k0c3 + 24), k034(k0c3 + 32), k035(k0c3 + 40), k036(k0c3 + 48), k037(k0c3 + 56);

                                        float32x4x2 k100(k1c0), k101(k1c0 + 8), k102(k1c0 + 16), k103(k1c0 + 24), k104(k1c0 + 32), k105(k1c0 + 40), k106(k1c0 + 48), k107(k1c0 + 56);
                                        float32x4x2 k110(k1c1), k111(k1c1 + 8), k112(k1c1 + 16), k113(k1c1 + 24), k114(k1c1 + 32), k115(k1c1 + 40), k116(k1c1 + 48), k117(k1c1 + 56);
                                        float32x4x2 k120(k1c2), k121(k1c2 + 8), k122(k1c2 + 16), k123(k1c2 + 24), k124(k1c2 + 32), k125(k1c2 + 40), k126(k1c2 + 48), k127(k1c2 + 56);
                                        float32x4x2 k130(k1c3), k131(k1c3 + 8), k132(k1c3 + 16), k133(k1c3 + 24), k134(k1c3 + 32), k135(k1c3 + 40), k136(k1c3 + 48), k137(k1c3 + 56);

                                        float32x4x2 k200(k2c0), k201(k2c0 + 8), k202(k2c0 + 16), k203(k2c0 + 24), k204(k2c0 + 32), k205(k2c0 + 40), k206(k2c0 + 48), k207(k2c0 + 56);
                                        float32x4x2 k210(k2c1), k211(k2c1 + 8), k212(k2c1 + 16), k213(k2c1 + 24), k214(k2c1 + 32), k215(k2c1 + 40), k216(k2c1 + 48), k217(k2c1 + 56);
                                        float32x4x2 k220(k2c2), k221(k2c2 + 8), k222(k2c2 + 16), k223(k2c2 + 24), k224(k2c2 + 32), k225(k2c2 + 40), k226(k2c2 + 48), k227(k2c2 + 56);
                                        float32x4x2 k230(k2c3), k231(k2c3 + 8), k232(k2c3 + 16), k233(k2c3 + 24), k234(k2c3 + 32), k235(k2c3 + 40), k236(k2c3 + 48), k237(k2c3 + 56);

                                        float32x4x2 k300(k3c0), k301(k3c0 + 8), k302(k3c0 + 16), k303(k3c0 + 24), k304(k3c0 + 32), k305(k3c0 + 40), k306(k3c0 + 48), k307(k3c0 + 56);
                                        float32x4x2 k310(k3c1), k311(k3c1 + 8), k312(k3c1 + 16), k313(k3c1 + 24), k314(k3c1 + 32), k315(k3c1 + 40), k316(k3c1 + 48), k317(k3c1 + 56);
                                        float32x4x2 k320(k3c2), k321(k3c2 + 8), k322(k3c2 + 16), k323(k3c2 + 24), k324(k3c2 + 32), k325(k3c2 + 40), k326(k3c2 + 48), k327(k3c2 + 56);
                                        float32x4x2 k330(k3c3), k331(k3c3 + 8), k332(k3c3 + 16), k333(k3c3 + 24), k334(k3c3 + 32), k335(k3c3 + 40), k336(k3c3 + 48), k337(k3c3 + 56);

                                        sum00 = fmadd(r00, k000, sum00); sum00 = fmadd(r10, k010, sum00); sum00 = fmadd(r20, k020, sum00); sum00 = fmadd(r30, k030, sum00);
                                        sum01 = fmadd(r01, k001, sum01); sum01 = fmadd(r11, k011, sum01); sum01 = fmadd(r21, k021, sum01); sum01 = fmadd(r31, k031, sum01);
                                        sum02 = fmadd(r02, k002, sum02); sum02 = fmadd(r12, k012, sum02); sum02 = fmadd(r22, k022, sum02); sum02 = fmadd(r32, k032, sum02);
                                        sum03 = fmadd(r03, k003, sum03); sum03 = fmadd(r13, k013, sum03); sum03 = fmadd(r23, k023, sum03); sum03 = fmadd(r33, k033, sum03);
                                        sum04 = fmadd(r04, k004, sum04); sum04 = fmadd(r14, k014, sum04); sum04 = fmadd(r24, k024, sum04); sum04 = fmadd(r34, k034, sum04);
                                        sum05 = fmadd(r05, k005, sum05); sum05 = fmadd(r15, k015, sum05); sum05 = fmadd(r25, k025, sum05); sum05 = fmadd(r35, k035, sum05);
                                        sum06 = fmadd(r06, k006, sum06); sum06 = fmadd(r16, k016, sum06); sum06 = fmadd(r26, k026, sum06); sum06 = fmadd(r36, k036, sum06);
                                        sum07 = fmadd(r07, k007, sum07); sum07 = fmadd(r17, k017, sum07); sum07 = fmadd(r27, k027, sum07); sum07 = fmadd(r37, k037, sum07);

                                        sum10 = fmadd(r00, k100, sum10); sum10 = fmadd(r10, k110, sum10); sum10 = fmadd(r20, k120, sum10); sum10 = fmadd(r30, k130, sum10);
                                        sum11 = fmadd(r01, k101, sum11); sum11 = fmadd(r11, k111, sum11); sum11 = fmadd(r21, k121, sum11); sum11 = fmadd(r31, k131, sum11);
                                        sum12 = fmadd(r02, k102, sum12); sum12 = fmadd(r12, k112, sum12); sum12 = fmadd(r22, k122, sum12); sum12 = fmadd(r32, k132, sum12);
                                        sum13 = fmadd(r03, k103, sum13); sum13 = fmadd(r13, k113, sum13); sum13 = fmadd(r23, k123, sum13); sum13 = fmadd(r33, k133, sum13);
                                        sum14 = fmadd(r04, k104, sum14); sum14 = fmadd(r14, k114, sum14); sum14 = fmadd(r24, k124, sum14); sum14 = fmadd(r34, k134, sum14);
                                        sum15 = fmadd(r05, k105, sum15); sum15 = fmadd(r15, k115, sum15); sum15 = fmadd(r25, k125, sum15); sum15 = fmadd(r35, k135, sum15);
                                        sum16 = fmadd(r06, k106, sum16); sum16 = fmadd(r16, k116, sum16); sum16 = fmadd(r26, k126, sum16); sum16 = fmadd(r36, k136, sum16);
                                        sum17 = fmadd(r07, k107, sum17); sum17 = fmadd(r17, k117, sum17); sum17 = fmadd(r27, k127, sum17); sum17 = fmadd(r37, k137, sum17);

                                        sum20 = fmadd(r00, k200, sum20); sum20 = fmadd(r10, k210, sum20); sum20 = fmadd(r20, k220, sum20); sum20 = fmadd(r30, k230, sum20);
                                        sum21 = fmadd(r01, k201, sum21); sum21 = fmadd(r11, k211, sum21); sum21 = fmadd(r21, k221, sum21); sum21 = fmadd(r31, k231, sum21);
                                        sum22 = fmadd(r02, k202, sum22); sum22 = fmadd(r12, k212, sum22); sum22 = fmadd(r22, k222, sum22); sum22 = fmadd(r32, k232, sum22);
                                        sum23 = fmadd(r03, k203, sum23); sum23 = fmadd(r13, k213, sum23); sum23 = fmadd(r23, k223, sum23); sum23 = fmadd(r33, k233, sum23);
                                        sum24 = fmadd(r04, k204, sum24); sum24 = fmadd(r14, k214, sum24); sum24 = fmadd(r24, k224, sum24); sum24 = fmadd(r34, k234, sum24);
                                        sum25 = fmadd(r05, k205, sum25); sum25 = fmadd(r15, k215, sum25); sum25 = fmadd(r25, k225, sum25); sum25 = fmadd(r35, k235, sum25);
                                        sum26 = fmadd(r06, k206, sum26); sum26 = fmadd(r16, k216, sum26); sum26 = fmadd(r26, k226, sum26); sum26 = fmadd(r36, k236, sum26);
                                        sum27 = fmadd(r07, k207, sum27); sum27 = fmadd(r17, k217, sum27); sum27 = fmadd(r27, k227, sum27); sum27 = fmadd(r37, k237, sum27);

                                        sum30 = fmadd(r00, k300, sum30); sum30 = fmadd(r10, k310, sum30); sum30 = fmadd(r20, k320, sum30); sum30 = fmadd(r30, k330, sum30);
                                        sum31 = fmadd(r01, k301, sum31); sum31 = fmadd(r11, k311, sum31); sum31 = fmadd(r21, k321, sum31); sum31 = fmadd(r31, k331, sum31);
                                        sum32 = fmadd(r02, k302, sum32); sum32 = fmadd(r12, k312, sum32); sum32 = fmadd(r22, k322, sum32); sum32 = fmadd(r32, k332, sum32);
                                        sum33 = fmadd(r03, k303, sum33); sum33 = fmadd(r13, k313, sum33); sum33 = fmadd(r23, k323, sum33); sum33 = fmadd(r33, k333, sum33);
                                        sum34 = fmadd(r04, k304, sum34); sum34 = fmadd(r14, k314, sum34); sum34 = fmadd(r24, k324, sum34); sum34 = fmadd(r34, k334, sum34);
                                        sum35 = fmadd(r05, k305, sum35); sum35 = fmadd(r15, k315, sum35); sum35 = fmadd(r25, k325, sum35); sum35 = fmadd(r35, k335, sum35);
                                        sum36 = fmadd(r06, k306, sum36); sum36 = fmadd(r16, k316, sum36); sum36 = fmadd(r26, k326, sum36); sum36 = fmadd(r36, k336, sum36);
                                        sum37 = fmadd(r07, k307, sum37); sum37 = fmadd(r17, k317, sum37); sum37 = fmadd(r27, k327, sum37); sum37 = fmadd(r37, k337, sum37);


                                    }

                                    for (int q = remain_inputch; q < input_channel; q++)
                                    {
                                        const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                                        const float* r0 = input_tm_at + i * 64;
                                        float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);

                                        const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                                        const float* k1 = kernel_tm_1 + q * ktm_c_offset;
                                        const float* k2 = kernel_tm_2 + q * ktm_c_offset;
                                        const float* k3 = kernel_tm_3 + q * ktm_c_offset;
                                        float32x4x2 k000(k0), k001(k0 + 8), k002(k0 + 16), k003(k0 + 24), k004(k0 + 32), k005(k0 + 40), k006(k0 + 48), k007(k0 + 56);
                                        float32x4x2 k100(k1), k101(k1 + 8), k102(k1 + 16), k103(k1 + 24), k104(k1 + 32), k105(k1 + 40), k106(k1 + 48), k107(k1 + 56);
                                        float32x4x2 k200(k2), k201(k2 + 8), k202(k2 + 16), k203(k2 + 24), k204(k2 + 32), k205(k2 + 40), k206(k2 + 48), k207(k2 + 56);
                                        float32x4x2 k300(k3), k301(k3 + 8), k302(k3 + 16), k303(k3 + 24), k304(k3 + 32), k305(k3 + 40), k306(k3 + 48), k307(k3 + 56);

                                        sum00 = fmadd(r00, k000, sum00);
                                        sum01 = fmadd(r01, k001, sum01);
                                        sum02 = fmadd(r02, k002, sum02);
                                        sum03 = fmadd(r03, k003, sum03);
                                        sum04 = fmadd(r04, k004, sum04);
                                        sum05 = fmadd(r05, k005, sum05);
                                        sum06 = fmadd(r06, k006, sum06);
                                        sum07 = fmadd(r07, k007, sum07);

                                        sum10 = fmadd(r00, k100, sum10);
                                        sum11 = fmadd(r01, k101, sum11);
                                        sum12 = fmadd(r02, k102, sum12);
                                        sum13 = fmadd(r03, k103, sum13);
                                        sum14 = fmadd(r04, k104, sum14);
                                        sum15 = fmadd(r05, k105, sum15);
                                        sum16 = fmadd(r06, k106, sum16);
                                        sum17 = fmadd(r07, k107, sum17);

                                        sum20 = fmadd(r00, k200, sum20);
                                        sum21 = fmadd(r01, k201, sum21);
                                        sum22 = fmadd(r02, k202, sum22);
                                        sum23 = fmadd(r03, k203, sum23);
                                        sum24 = fmadd(r04, k204, sum24);
                                        sum25 = fmadd(r05, k205, sum25);
                                        sum26 = fmadd(r06, k206, sum26);
                                        sum27 = fmadd(r07, k207, sum27);

                                        sum30 = fmadd(r00, k300, sum30);
                                        sum31 = fmadd(r01, k301, sum31);
                                        sum32 = fmadd(r02, k302, sum32);
                                        sum33 = fmadd(r03, k303, sum33);
                                        sum34 = fmadd(r04, k304, sum34);
                                        sum35 = fmadd(r05, k305, sum35);
                                        sum36 = fmadd(r06, k306, sum36);
                                        sum37 = fmadd(r07, k307, sum37);

                                    }

                                    sum00.store(out_0); sum01.store(out_0 + 8); sum02.store(out_0 + 16); sum03.store(out_0 + 24);
                                    sum04.store(out_0 + 32); sum05.store(out_0 + 40); sum06.store(out_0 + 48); sum07.store(out_0 + 56);

                                    sum10.store(out_1); sum11.store(out_1 + 8); sum12.store(out_1 + 16); sum13.store(out_1 + 24);
                                    sum14.store(out_1 + 32); sum15.store(out_1 + 40); sum16.store(out_1 + 48); sum17.store(out_1 + 56);

                                    sum20.store(out_2); sum21.store(out_2 + 8); sum22.store(out_2 + 16); sum23.store(out_2 + 24);
                                    sum24.store(out_2 + 32); sum25.store(out_2 + 40); sum26.store(out_2 + 48); sum27.store(out_2 + 56);

                                    sum30.store(out_3); sum31.store(out_3 + 8); sum32.store(out_3 + 16); sum33.store(out_3 + 24);
                                    sum34.store(out_3 + 32); sum35.store(out_3 + 40); sum36.store(out_3 + 48); sum37.store(out_3 + 56);

                                }
                                out_tm_0 += (4 * outtm_c_offset);
                                kernel_tm_0 += (4 * ktm_n_offset);
                            }

                        });
                    }
                    thread_pool->join();
                }


                for (int c = remain_outch; c < output_channel; c++)
                {
                    float* out_tm_0 = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;

                    const float* kernel_tm_ptr = k_tm.data<float>();
                    const float* kernel_tm_0 = kernel_tm_ptr + c * ktm_n_offset;

                    for (int i = 0; i < num_blocks; i++)
                    {
                        float* out_0 = out_tm_0 + i * 64;

                        float32x4x2 sum00(0.f), sum01(0.f), sum02(0.f), sum03(0.f), sum04(0.f), sum05(0.f), sum06(0.f), sum07(0.f);

                        int q = 0;
                        for (; q + 3 < input_channel; q += 4)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 64;
                            const float* r1 = r0 + tm_c_offset;
                            const float* r2 = r1 + tm_c_offset;
                            const float* r3 = r2 + tm_c_offset;
                            float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);
                            float32x4x2 r10(r1), r11(r1 + 8), r12(r1 + 16), r13(r1 + 24), r14(r1 + 32), r15(r1 + 40), r16(r1 + 48), r17(r1 + 56);
                            float32x4x2 r20(r2), r21(r2 + 8), r22(r2 + 16), r23(r2 + 24), r24(r2 + 32), r25(r2 + 40), r26(r2 + 48), r27(r2 + 56);
                            float32x4x2 r30(r3), r31(r3 + 8), r32(r3 + 16), r33(r3 + 24), r34(r3 + 32), r35(r3 + 40), r36(r3 + 48), r37(r3 + 56);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            const float* k1 = k0 + ktm_c_offset;
                            const float* k2 = k1 + ktm_c_offset;
                            const float* k3 = k2 + ktm_c_offset;

                            float32x4x2 k000(k0), k001(k0 + 8), k002(k0 + 16), k003(k0 + 24), k004(k0 + 32), k005(k0 + 40), k006(k0 + 48), k007(k0 + 56);
                            float32x4x2 k010(k1), k011(k1 + 8), k012(k1 + 16), k013(k1 + 24), k014(k1 + 32), k015(k1 + 40), k016(k1 + 48), k017(k1 + 56);
                            float32x4x2 k020(k2), k021(k2 + 8), k022(k2 + 16), k023(k2 + 24), k024(k2 + 32), k025(k2 + 40), k026(k2 + 48), k027(k2 + 56);
                            float32x4x2 k030(k3), k031(k3 + 8), k032(k3 + 16), k033(k3 + 24), k034(k3 + 32), k035(k3 + 40), k036(k3 + 48), k037(k3 + 56);

                            sum00 = fmadd(r00, k000, sum00); sum00 = fmadd(r10, k010, sum00); sum00 = fmadd(r20, k020, sum00); sum00 = fmadd(r30, k030, sum00);
                            sum01 = fmadd(r01, k001, sum01); sum01 = fmadd(r11, k011, sum01); sum01 = fmadd(r21, k021, sum01); sum01 = fmadd(r31, k031, sum01);
                            sum02 = fmadd(r02, k002, sum02); sum02 = fmadd(r12, k012, sum02); sum02 = fmadd(r22, k022, sum02); sum02 = fmadd(r32, k032, sum02);
                            sum03 = fmadd(r03, k003, sum03); sum03 = fmadd(r13, k013, sum03); sum03 = fmadd(r23, k023, sum03); sum03 = fmadd(r33, k033, sum03);
                            sum04 = fmadd(r04, k004, sum04); sum04 = fmadd(r14, k014, sum04); sum04 = fmadd(r24, k024, sum04); sum04 = fmadd(r34, k034, sum04);
                            sum05 = fmadd(r05, k005, sum05); sum05 = fmadd(r15, k015, sum05); sum05 = fmadd(r25, k025, sum05); sum05 = fmadd(r35, k035, sum05);
                            sum06 = fmadd(r06, k006, sum06); sum06 = fmadd(r16, k016, sum06); sum06 = fmadd(r26, k026, sum06); sum06 = fmadd(r36, k036, sum06);
                            sum07 = fmadd(r07, k007, sum07); sum07 = fmadd(r17, k017, sum07); sum07 = fmadd(r27, k027, sum07); sum07 = fmadd(r37, k037, sum07);
                        }

                        for (; q < input_channel; q++)
                        {
                            const float* input_tm_at = input_tm_ptr + n * tm_num_offset + q * tm_c_offset;
                            const float* r0 = input_tm_at + i * 64;
                            float32x4x2 r00(r0), r01(r0 + 8), r02(r0 + 16), r03(r0 + 24), r04(r0 + 32), r05(r0 + 40), r06(r0 + 48), r07(r0 + 56);

                            const float* k0 = kernel_tm_0 + q * ktm_c_offset;
                            float32x4x2 k000(k0), k001(k0 + 8), k002(k0 + 16), k003(k0 + 24), k004(k0 + 32), k005(k0 + 40), k006(k0 + 48), k007(k0 + 56);

                            sum00 = fmadd(r00, k000, sum00);
                            sum01 = fmadd(r01, k001, sum01);
                            sum02 = fmadd(r02, k002, sum02);
                            sum03 = fmadd(r03, k003, sum03);
                            sum04 = fmadd(r04, k004, sum04);
                            sum05 = fmadd(r05, k005, sum05);
                            sum06 = fmadd(r06, k006, sum06);
                            sum07 = fmadd(r07, k007, sum07);

                        }

                        sum00.store(out_0); sum01.store(out_0 + 8); sum02.store(out_0 + 16); sum03.store(out_0 + 24);
                        sum04.store(out_0 + 32); sum05.store(out_0 + 40); sum06.store(out_0 + 48); sum07.store(out_0 + 56);

                    }
                }
            }


            //begin transform output
            Shape output_bordered_s = { num, output_channel, output_h, output_w };
            Tensor output_bordered(MemoryDevice(CPU), out.dtype(), output_bordered_s);
            int outbo_c_offset = output_h * output_w;
            int outbo_n_offset = output_channel * outbo_c_offset;

            float* out_ptr = output_bordered.data<float>();

            //const float AT[6][8] = {
            //    {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
            //    {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
            //    {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
            //};

            // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
            // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
            // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
            // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
            // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
            // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

            //reuse (r1 + r2) (r1 - r2) (r3 + r4) (r3 - r4) (r5 + r6) (r5 - r6)

            for (int n = 0; n < num; n++)
            {
                if (thread_pool == nullptr || thread_pool->size() <= 1)
                {
                    for (int c = 0; c < output_channel; c++)
                    {
                        float* output_tm_at = out_tm_ptr + n * outtm_n_offset + c * outtm_c_offset;
                        float* out_at = out_ptr + n * outbo_n_offset + c * outbo_c_offset;

                        float32x4 tmp1add2, tmp1sub2;
                        float32x4 tmp3add4, tmp3sub4;
                        float32x4 tmp5add6, tmp5sub6;

                        const float32x4 f0(0.0f), f2(2.0f), f4(4.0f);
                        const float32x4 f8(8.0f), f16(16.0f), f32(32.0f);

                        for (int i = 0; i < col_blocks; i++)
                        {
                            for (int j = 0; j < row_blocks; j++)
                            {
                                const float* w0 = output_tm_at + (i * col_blocks + j) * 64;
                                const float* w1 = w0 + 8;
                                const float* w2 = w1 + 8;
                                const float* w3 = w2 + 8;
                                const float* w4 = w3 + 8;
                                const float* w5 = w4 + 8;
                                const float* w6 = w5 + 8;
                                const float* w7 = w6 + 8;

                                float32x4 l0(w0), r0(w0 + 4), l1(w1), r1(w1 + 4);
                                float32x4 l2(w2), r2(w2 + 4), l3(w3), r3(w3 + 4);
                                float32x4 l4(w4), r4(w4 + 4), l5(w5), r5(w5 + 4);
                                float32x4 l6(w6), r6(w6 + 4), l7(w7), r7(w7 + 4);

                                winograd_f63_output_transform(l0, l1, l2, l3, l4, l5, l6, l7,
                                    tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                    f0, f2, f4, f8, f16, f32);
                                transposex4x4(l0, l1, l2, l3);
                                transposex4x4(l4, l5, l6, l7);

                                winograd_f63_output_transform(r0, r1, r2, r3, r4, r5, r6, r7,
                                    tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                    f0, f2, f4, f8, f16, f32);
                                transposex4x4(r0, r1, r2, r3);
                                transposex4x4(r4, r5, r6, r7);

                                winograd_f63_output_transform(l0, l1, l2, l3, r0, r1, r2, r3,
                                    tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                    f0, f2, f4, f8, f16, f32);

                                winograd_f63_output_transform(l4, l5, l6, l7, r4, r5, r6, r7,
                                    tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                    f0, f2, f4, f8, f16, f32);

                                float* d0 = out_at + i * output_w * 6 + j * 6;
                                float* d1 = d0 + output_w;
                                float* d2 = d1 + output_w;
                                float* d3 = d2 + output_w;
                                float* d4 = d3 + output_w;
                                float* d5 = d4 + output_w;

                                if (((j * 6 + 6) > output_h) || ((i * 6 + 6) > output_w)) {
                                    l0.store(d0); l4.store(d0 + 4);
                                    l1.store(d1); l5.store(d1 + 4);
                                    l2.store(d2); l6.store(d2 + 4);
                                    l3.store(d3); l7.store(d3 + 4);

                                    r0.store(d4); r4.store(d4 + 4);
                                    r1.store(d5); r5.store(d5 + 4);
                                }
                                else {
                                    l0.store(d0); *(d0 + 4) = *(((float*)&(l4.value))); *(d0 + 5) = *(((float*)&(l4.value)) + 1);
                                    l1.store(d1); *(d1 + 4) = *(((float*)&(l5.value))); *(d1 + 5) = *(((float*)&(l5.value)) + 1);
                                    l2.store(d2); *(d2 + 4) = *(((float*)&(l6.value))); *(d2 + 5) = *(((float*)&(l6.value)) + 1);
                                    l3.store(d3); *(d3 + 4) = *(((float*)&(l7.value))); *(d3 + 5) = *(((float*)&(l7.value)) + 1);

                                    r0.store(d4); *(d4 + 4) = *(((float*)&(r4.value))); *(d4 + 5) = *(((float*)&(r4.value)) + 1);
                                    r1.store(d5); *(d5 + 4) = *(((float*)&(r5.value))); *(d5 + 5) = *(((float*)&(r5.value)) + 1);
                                }

                            }
                        }
                    }
                }
                else
                {
                    auto bins = split_bins(0, output_channel, (int)thread_pool->size());
                    for (auto &bin : bins)
                    {
                        thread_pool->run([&, n, out_tm_ptr, out_ptr, bin](int) {

                            float* output_tm_at = out_tm_ptr + n * outtm_n_offset + bin.first * outtm_c_offset;
                            float* out_at = out_ptr + n * outbo_n_offset + bin.first * outbo_c_offset;

                            for (int c = bin.first; c < bin.second; c++)
                            {
                                float32x4 tmp1add2, tmp1sub2;
                                float32x4 tmp3add4, tmp3sub4;
                                float32x4 tmp5add6, tmp5sub6;

                                const float32x4 f0(0.0f), f2(2.0f), f4(4.0f);
                                const float32x4 f8(8.0f), f16(16.0f), f32(32.0f);

                                for (int i = 0; i < col_blocks; i++)
                                {
                                    for (int j = 0; j < row_blocks; j++)
                                    {
                                        const float* w0 = output_tm_at + (i * col_blocks + j) * 64;
                                        const float* w1 = w0 + 8;
                                        const float* w2 = w1 + 8;
                                        const float* w3 = w2 + 8;
                                        const float* w4 = w3 + 8;
                                        const float* w5 = w4 + 8;
                                        const float* w6 = w5 + 8;
                                        const float* w7 = w6 + 8;

                                        float32x4 l0(w0), r0(w0 + 4), l1(w1), r1(w1 + 4);
                                        float32x4 l2(w2), r2(w2 + 4), l3(w3), r3(w3 + 4);
                                        float32x4 l4(w4), r4(w4 + 4), l5(w5), r5(w5 + 4);
                                        float32x4 l6(w6), r6(w6 + 4), l7(w7), r7(w7 + 4);

                                        winograd_f63_output_transform(l0, l1, l2, l3, l4, l5, l6, l7,
                                            tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                            f0, f2, f4, f8, f16, f32);
                                        transposex4x4(l0, l1, l2, l3);
                                        transposex4x4(l4, l5, l6, l7);

                                        winograd_f63_output_transform(r0, r1, r2, r3, r4, r5, r6, r7,
                                            tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                            f0, f2, f4, f8, f16, f32);
                                        transposex4x4(r0, r1, r2, r3);
                                        transposex4x4(r4, r5, r6, r7);

                                        winograd_f63_output_transform(l0, l1, l2, l3, r0, r1, r2, r3,
                                            tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                            f0, f2, f4, f8, f16, f32);

                                        winograd_f63_output_transform(l4, l5, l6, l7, r4, r5, r6, r7,
                                            tmp1add2, tmp1sub2, tmp3add4, tmp3sub4, tmp5add6, tmp5sub6,
                                            f0, f2, f4, f8, f16, f32);

                                        float* d0 = out_at + i * output_w * 6 + j * 6;
                                        float* d1 = d0 + output_w;
                                        float* d2 = d1 + output_w;
                                        float* d3 = d2 + output_w;
                                        float* d4 = d3 + output_w;
                                        float* d5 = d4 + output_w;

                                        if (((j * 6 + 6) > output_h) || ((i * 6 + 6) > output_w)) {
                                            l0.store(d0); l4.store(d0 + 4);
                                            l1.store(d1); l5.store(d1 + 4);
                                            l2.store(d2); l6.store(d2 + 4);
                                            l3.store(d3); l7.store(d3 + 4);

                                            r0.store(d4); r4.store(d4 + 4);
                                            r1.store(d5); r5.store(d5 + 4);
                                        }
                                        else {
                                            l0.store(d0); *(d0 + 4) = *(((float*)&(l4.value))); *(d0 + 5) = *(((float*)&(l4.value)) + 1);
                                            l1.store(d1); *(d1 + 4) = *(((float*)&(l5.value))); *(d1 + 5) = *(((float*)&(l5.value)) + 1);
                                            l2.store(d2); *(d2 + 4) = *(((float*)&(l6.value))); *(d2 + 5) = *(((float*)&(l6.value)) + 1);
                                            l3.store(d3); *(d3 + 4) = *(((float*)&(l7.value))); *(d3 + 5) = *(((float*)&(l7.value)) + 1);

                                            r0.store(d4); *(d4 + 4) = *(((float*)&(r4.value))); *(d4 + 5) = *(((float*)&(r4.value)) + 1);
                                            r1.store(d5); *(d5 + 4) = *(((float*)&(r5.value))); *(d5 + 5) = *(((float*)&(r5.value)) + 1);
                                        }

                                    }
                                }
                                output_tm_at += outtm_c_offset;
                                out_at += outbo_c_offset;
                            }
                        });
                    }
                    thread_pool->join();
                }
            }

            inner_cut<float>(output_bordered, out, 0, output_h - out_shape[2], 0, output_w - out_shape[3]);

        }
#endif
        template<typename T>
        void Conv2dAlgorithm<T>::conv2d_3x3_sse(const Tensor &x, const Tensor &w, Tensor &out) {

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const T *pinput = x.data<T>();
            const T *pweight = w.data<T>();
            T *poutput = out.data<T>();
            std::memset(poutput, 0, sizeof(T)*out.count());

            for (int n = 0; n < number; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int outc_index = 0; outc_index < out_channel; outc_index++)
                {
                    T* out_at = poutput + n*out_num_offset + outc_index * out_channel_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        T* out_ptr_0 = out_at;
                        T* out_ptr_1 = out_ptr_0 + out_width;

                        const T* input_cur = pinput + n*input_num_offset + inc_index*input_channel_offset;

                        const T* kernel_cur = pweight + outc_index * input_channel * 9 + inc_index * 9;

                        const T* r_0 = input_cur;
                        const T* r_1 = input_cur + input_width;
                        const T* r_2 = input_cur + input_width * 2;
                        const T* r_3 = input_cur + input_width * 3;

                        const T* k_0 = kernel_cur;
                        const T* k_1 = kernel_cur + 3;
                        const T* k_2 = kernel_cur + 6;

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_0 = 0, sum_1 = 0;

                                sum_0 += r_0[0] * k_0[0];
                                sum_0 += r_0[1] * k_0[1];
                                sum_0 += r_0[2] * k_0[2];
                                sum_0 += r_1[0] * k_1[0];
                                sum_0 += r_1[1] * k_1[1];
                                sum_0 += r_1[2] * k_1[2];
                                sum_0 += r_2[0] * k_2[0];
                                sum_0 += r_2[1] * k_2[1];
                                sum_0 += r_2[2] * k_2[2];

                                sum_1 += r_1[0] * k_0[0];
                                sum_1 += r_1[1] * k_0[1];
                                sum_1 += r_1[2] * k_0[2];
                                sum_1 += r_2[0] * k_1[0];
                                sum_1 += r_2[1] * k_1[1];
                                sum_1 += r_2[2] * k_1[2];
                                sum_1 += r_3[0] * k_2[0];
                                sum_1 += r_3[1] * k_2[1];
                                sum_1 += r_3[2] * k_2[2];

                                *out_ptr_0 += sum_0;
                                *out_ptr_1 += sum_1;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum = 0;

                                sum += r_0[0] * k_0[0];
                                sum += r_0[1] * k_0[1];
                                sum += r_0[2] * k_0[2];
                                sum += r_1[0] * k_1[0];
                                sum += r_1[1] * k_1[1];
                                sum += r_1[2] * k_1[2];
                                sum += r_2[0] * k_2[0];
                                sum += r_2[1] * k_2[1];
                                sum += r_2[2] * k_2[2];

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }
                    }
                }
            }
        }

        template<>
        void Conv2dAlgorithm<float>::conv2d_3x3_sse(const Tensor &x, const Tensor &w, Tensor &out) {

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const float *pinput = x.data<float>();
            const float *pweight = w.data<float>();
            float *poutput = out.data<float>();
            std::memset(poutput, 0, sizeof(float)*out.count());

            for (int n = 0; n < number; n++)
            {
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int outc_index = 0; outc_index < out_channel; outc_index++)
                {
                    float* out_at = poutput + n*out_num_offset + outc_index * out_channel_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        float* out_ptr_0 = out_at;
                        float* out_ptr_1 = out_ptr_0 + out_width;

                        const float* input_cur = pinput + n*input_num_offset + inc_index*input_channel_offset;

                        const float* kernel_cur = pweight + outc_index * input_channel * 9 + inc_index * 9;

                        const float* r_0 = input_cur;
                        const float* r_1 = input_cur + input_width;
                        const float* r_2 = input_cur + input_width * 2;
                        const float* r_3 = input_cur + input_width * 3;

                        const float* k_0 = kernel_cur;
                        const float* k_1 = kernel_cur + 3;
                        const float* k_2 = kernel_cur + 6;
                        float32x4 k00(k_0);
                        float32x4 k10(k_1);
                        float32x4 k20(k_2);

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);
                                float32x4 r30(r_3);

                                float32x4 sum00(float(0));

                                sum00 += r00 * k00;
                                sum00 += r10 * k10;
                                sum00 += r20 * k20;

                                float32x4 sum10(float(0));
                                sum10 += r10 * k00;
                                sum10 += r20 * k10;
                                sum10 += r30 * k20;

                                float sum0 = ts::sum(sum00, 3);
                                float sum1 = ts::sum(sum10, 3);

                                *out_ptr_0 += sum0;
                                *out_ptr_1 += sum1;

                                //sum00.store(out_ptr_0);
                                //sum10.store(out_ptr_1);

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);

                                float32x4 sum_x4(float(0));

                                sum_x4 += r00 * k00;
                                sum_x4 += r10 * k10;
                                sum_x4 += r20 * k20;

                                float sum = ts::sum(sum_x4, 3);

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }
                    }
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::conv2d_3x3_sse_inplace(const Tensor &x, const Tensor &w, Tensor &out) {

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const T *pinput = x.data<T>();
            const T *pweight = w.data<T>();
            T *poutput = out.data<T>();
            std::memset(poutput, 0, sizeof(T)*out.count());

            for (int n = 0; n < number; n++)
            {
                int for_out_channel = out_channel >> 2;
                int remain_out_channel = for_out_channel << 2;
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int outc_index = 0; outc_index < for_out_channel; outc_index++)
                {
                    int p = outc_index * 4;
                    T *out_at = poutput + n * out_num_offset + p * out_channel_offset;
                    T *out_ptr_0 = out_at;
                    T *out_ptr_1 = out_ptr_0 + out_channel_offset;
                    T *out_ptr_2 = out_ptr_1 + out_channel_offset;
                    T *out_ptr_3 = out_ptr_2 + out_channel_offset;

                    int kernel_num_offset = input_channel * 9;
                    const T* kernel_cur = pweight + p * kernel_num_offset;
                    const T* k_0 = kernel_cur;
                    const T* k_1 = k_0 + kernel_num_offset;
                    const T* k_2 = k_1 + kernel_num_offset;
                    const T* k_3 = k_2 + kernel_num_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        T* out_ptr_00 = out_ptr_0;
                        T* out_ptr_01 = out_ptr_0 + out_width;
                        T* out_ptr_10 = out_ptr_1;
                        T* out_ptr_11 = out_ptr_1 + out_width;
                        T* out_ptr_20 = out_ptr_2;
                        T* out_ptr_21 = out_ptr_2 + out_width;
                        T* out_ptr_30 = out_ptr_3;
                        T* out_ptr_31 = out_ptr_3 + out_width;

                        const T* input_cur = pinput + n * input_num_offset + inc_index * input_channel_offset;

                        const T* r_0 = input_cur;
                        const T* r_1 = r_0 + input_width;
                        const T* r_2 = r_1 + input_width;
                        const T* r_3 = r_2 + input_width;

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_00 = 0, sum_01 = 0;
                                T sum_10 = 0, sum_11 = 0;
                                T sum_20 = 0, sum_21 = 0;
                                T sum_30 = 0, sum_31 = 0;

                                sum_00 += r_0[0] * k_0[0];
                                sum_00 += r_0[1] * k_0[1];
                                sum_00 += r_0[2] * k_0[2];
                                sum_00 += r_1[0] * k_0[3];
                                sum_00 += r_1[1] * k_0[4];
                                sum_00 += r_1[2] * k_0[5];
                                sum_00 += r_2[0] * k_0[6];
                                sum_00 += r_2[1] * k_0[7];
                                sum_00 += r_2[2] * k_0[8];

                                sum_01 += r_1[0] * k_0[0];
                                sum_01 += r_1[1] * k_0[1];
                                sum_01 += r_1[2] * k_0[2];
                                sum_01 += r_2[0] * k_0[3];
                                sum_01 += r_2[1] * k_0[4];
                                sum_01 += r_2[2] * k_0[5];
                                sum_01 += r_3[0] * k_0[6];
                                sum_01 += r_3[1] * k_0[7];
                                sum_01 += r_3[2] * k_0[8];

                                sum_10 += r_0[0] * k_1[0];
                                sum_10 += r_0[1] * k_1[1];
                                sum_10 += r_0[2] * k_1[2];
                                sum_10 += r_1[0] * k_1[3];
                                sum_10 += r_1[1] * k_1[4];
                                sum_10 += r_1[2] * k_1[5];
                                sum_10 += r_2[0] * k_1[6];
                                sum_10 += r_2[1] * k_1[7];
                                sum_10 += r_2[2] * k_1[8];

                                sum_11 += r_1[0] * k_1[0];
                                sum_11 += r_1[1] * k_1[1];
                                sum_11 += r_1[2] * k_1[2];
                                sum_11 += r_2[0] * k_1[3];
                                sum_11 += r_2[1] * k_1[4];
                                sum_11 += r_2[2] * k_1[5];
                                sum_11 += r_3[0] * k_1[6];
                                sum_11 += r_3[1] * k_1[7];
                                sum_11 += r_3[2] * k_1[8];

                                sum_20 += r_0[0] * k_2[0];
                                sum_20 += r_0[1] * k_2[1];
                                sum_20 += r_0[2] * k_2[2];
                                sum_20 += r_1[0] * k_2[3];
                                sum_20 += r_1[1] * k_2[4];
                                sum_20 += r_1[2] * k_2[5];
                                sum_20 += r_2[0] * k_2[6];
                                sum_20 += r_2[1] * k_2[7];
                                sum_20 += r_2[2] * k_2[8];

                                sum_21 += r_1[0] * k_2[0];
                                sum_21 += r_1[1] * k_2[1];
                                sum_21 += r_1[2] * k_2[2];
                                sum_21 += r_2[0] * k_2[3];
                                sum_21 += r_2[1] * k_2[4];
                                sum_21 += r_2[2] * k_2[5];
                                sum_21 += r_3[0] * k_2[6];
                                sum_21 += r_3[1] * k_2[7];
                                sum_21 += r_3[2] * k_2[8];

                                sum_30 += r_0[0] * k_3[0];
                                sum_30 += r_0[1] * k_3[1];
                                sum_30 += r_0[2] * k_3[2];
                                sum_30 += r_1[0] * k_3[3];
                                sum_30 += r_1[1] * k_3[4];
                                sum_30 += r_1[2] * k_3[5];
                                sum_30 += r_2[0] * k_3[6];
                                sum_30 += r_2[1] * k_3[7];
                                sum_30 += r_2[2] * k_3[8];

                                sum_31 += r_1[0] * k_3[0];
                                sum_31 += r_1[1] * k_3[1];
                                sum_31 += r_1[2] * k_3[2];
                                sum_31 += r_2[0] * k_3[3];
                                sum_31 += r_2[1] * k_3[4];
                                sum_31 += r_2[2] * k_3[5];
                                sum_31 += r_3[0] * k_3[6];
                                sum_31 += r_3[1] * k_3[7];
                                sum_31 += r_3[2] * k_3[8];

                                *out_ptr_00 += sum_00;
                                *out_ptr_01 += sum_01;
                                *out_ptr_10 += sum_10;
                                *out_ptr_11 += sum_11;
                                *out_ptr_20 += sum_20;
                                *out_ptr_21 += sum_21;
                                *out_ptr_30 += sum_30;
                                *out_ptr_31 += sum_31;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_01++;
                                out_ptr_10++;
                                out_ptr_11++;
                                out_ptr_20++;
                                out_ptr_21++;
                                out_ptr_30++;
                                out_ptr_31++;

                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_00 += out_width;
                            out_ptr_01 += out_width;
                            out_ptr_10 += out_width;
                            out_ptr_11 += out_width;
                            out_ptr_20 += out_width;
                            out_ptr_21 += out_width;
                            out_ptr_30 += out_width;
                            out_ptr_31 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_0 = 0, sum_1 = 0, sum_2 = 0, sum_3 = 0;

                                sum_0 += r_0[0] * k_0[0];
                                sum_0 += r_0[1] * k_0[1];
                                sum_0 += r_0[2] * k_0[2];
                                sum_0 += r_1[0] * k_0[3];
                                sum_0 += r_1[1] * k_0[4];
                                sum_0 += r_1[2] * k_0[5];
                                sum_0 += r_2[0] * k_0[6];
                                sum_0 += r_2[1] * k_0[7];
                                sum_0 += r_2[2] * k_0[8];

                                sum_1 += r_0[0] * k_1[0];
                                sum_1 += r_0[1] * k_1[1];
                                sum_1 += r_0[2] * k_1[2];
                                sum_1 += r_1[0] * k_1[3];
                                sum_1 += r_1[1] * k_1[4];
                                sum_1 += r_1[2] * k_1[5];
                                sum_1 += r_2[0] * k_1[6];
                                sum_1 += r_2[1] * k_1[7];
                                sum_1 += r_2[2] * k_1[8];

                                sum_2 += r_0[0] * k_2[0];
                                sum_2 += r_0[1] * k_2[1];
                                sum_2 += r_0[2] * k_2[2];
                                sum_2 += r_1[0] * k_2[3];
                                sum_2 += r_1[1] * k_2[4];
                                sum_2 += r_1[2] * k_2[5];
                                sum_2 += r_2[0] * k_2[6];
                                sum_2 += r_2[1] * k_2[7];
                                sum_2 += r_2[2] * k_2[8];

                                sum_3 += r_0[0] * k_3[0];
                                sum_3 += r_0[1] * k_3[1];
                                sum_3 += r_0[2] * k_3[2];
                                sum_3 += r_1[0] * k_3[3];
                                sum_3 += r_1[1] * k_3[4];
                                sum_3 += r_1[2] * k_3[5];
                                sum_3 += r_2[0] * k_3[6];
                                sum_3 += r_2[1] * k_3[7];
                                sum_3 += r_2[2] * k_3[8];

                                *out_ptr_00 += sum_0;
                                *out_ptr_10 += sum_1;
                                *out_ptr_20 += sum_2;
                                *out_ptr_30 += sum_3;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_10++;
                                out_ptr_20++;
                                out_ptr_30++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        k_0 += 9;
                        k_1 += 9;
                        k_2 += 9;
                        k_3 += 9;
                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int p = remain_out_channel; p < out_channel; p++)
                {
                    T *out_at = poutput + n * out_num_offset + p * out_channel_offset;

                    const T* kernel_cur = pweight + p * input_channel * 9;

                    for (int q = 0; q < input_channel; q++)
                    {
                        T* out_ptr_0 = out_at;
                        T* out_ptr_1 = out_ptr_0 + out_width;

                        const T* input_cur = pinput + n * input_num_offset + q * input_channel_offset;

                        const T* r_0 = input_cur;
                        const T* r_1 = r_0 + input_width;
                        const T* r_2 = r_1 + input_width;
                        const T* r_3 = r_2 + input_width;

                        const T* k_0 = kernel_cur;
                        const T* k_1 = kernel_cur + 3;
                        const T* k_2 = kernel_cur + 6;

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum_0 = 0, sum_1 = 0;

                                sum_0 += r_0[0] * k_0[0];
                                sum_0 += r_0[1] * k_0[1];
                                sum_0 += r_0[2] * k_0[2];
                                sum_0 += r_1[0] * k_1[0];
                                sum_0 += r_1[1] * k_1[1];
                                sum_0 += r_1[2] * k_1[2];
                                sum_0 += r_2[0] * k_2[0];
                                sum_0 += r_2[1] * k_2[1];
                                sum_0 += r_2[2] * k_2[2];

                                sum_1 += r_1[0] * k_0[0];
                                sum_1 += r_1[1] * k_0[1];
                                sum_1 += r_1[2] * k_0[2];
                                sum_1 += r_2[0] * k_1[0];
                                sum_1 += r_2[1] * k_1[1];
                                sum_1 += r_2[2] * k_1[2];
                                sum_1 += r_3[0] * k_2[0];
                                sum_1 += r_3[1] * k_2[1];
                                sum_1 += r_3[2] * k_2[2];

                                *out_ptr_0 += sum_0;
                                *out_ptr_1 += sum_1;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                T sum = 0;

                                sum += r_0[0] * k_0[0];
                                sum += r_0[1] * k_0[1];
                                sum += r_0[2] * k_0[2];
                                sum += r_1[0] * k_1[0];
                                sum += r_1[1] * k_1[1];
                                sum += r_1[2] * k_1[2];
                                sum += r_2[0] * k_2[0];
                                sum += r_2[1] * k_2[1];
                                sum += r_2[2] * k_2[2];

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        kernel_cur += 9;
                    }
                }
            }
        }

        template<>
        void Conv2dAlgorithm<float>::conv2d_3x3_sse_inplace(const Tensor &x, const Tensor &w, Tensor &out) {

            auto x_shape = x.sizes();
            auto out_shape = out.sizes();
            int number = x_shape[0];
            int input_channel = x_shape[1];
            int input_height = x_shape[2];
            int input_width = x_shape[3];
            int input_channel_offset = input_height *input_width;
            int input_num_offset = input_channel * input_channel_offset;

            int out_channel = out_shape[1];
            int out_height = out_shape[2];
            int out_width = out_shape[3];

            int out_channel_offset = out_height * out_width;
            int out_num_offset = out_channel * out_channel_offset;

            const float *pinput = x.data<float>();
            const float *pweight = w.data<float>();
            float *poutput = out.data<float>();
            std::memset(poutput, 0, sizeof(float)*out.count());

            for (int n = 0; n < number; n++)
            {
                int for_out_channel = out_channel >> 2;
                int remain_out_channel = for_out_channel << 2;
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int outc_index = 0; outc_index < for_out_channel; outc_index++)
                {
                    int p = outc_index * 4;
                    float *out_at = poutput + n * out_num_offset + p * out_channel_offset;
                    float *out_ptr_0 = out_at;
                    float *out_ptr_1 = out_ptr_0 + out_channel_offset;
                    float *out_ptr_2 = out_ptr_1 + out_channel_offset;
                    float *out_ptr_3 = out_ptr_2 + out_channel_offset;

                    int kernel_num_offset = input_channel * 9;
                    const float* kernel_cur = pweight + p * kernel_num_offset;
                    const float* k_0 = kernel_cur;
                    const float* k_1 = k_0 + kernel_num_offset;
                    const float* k_2 = k_1 + kernel_num_offset;
                    const float* k_3 = k_2 + kernel_num_offset;

                    for (int inc_index = 0; inc_index < input_channel; inc_index++)
                    {
                        float32x4 k00(k_0);
                        float32x4 k01(k_0 + 3);
                        float32x4 k02(k_0 + 6);

                        float32x4 k10(k_1);
                        float32x4 k11(k_1 + 3);
                        float32x4 k12(k_1 + 6);

                        float32x4 k20(k_2);
                        float32x4 k21(k_2 + 3);
                        float32x4 k22(k_2 + 6);

                        float32x4 k30(k_3);
                        float32x4 k31(k_3 + 3);
                        float32x4 k32(k_3 + 6);

                        float* out_ptr_00 = out_ptr_0;
                        float* out_ptr_01 = out_ptr_0 + out_width;
                        float* out_ptr_10 = out_ptr_1;
                        float* out_ptr_11 = out_ptr_1 + out_width;
                        float* out_ptr_20 = out_ptr_2;
                        float* out_ptr_21 = out_ptr_2 + out_width;
                        float* out_ptr_30 = out_ptr_3;
                        float* out_ptr_31 = out_ptr_3 + out_width;

                        const float* input_cur = pinput + n * input_num_offset + inc_index * input_channel_offset;

                        const float* r_0 = input_cur;
                        const float* r_1 = r_0 + input_width;
                        const float* r_2 = r_1 + input_width;
                        const float* r_3 = r_2 + input_width;

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r01(r_1);
                                float32x4 r02(r_2);
                                float32x4 r03(r_3);

                                float32x4 sum_00_x4(float(0)); float32x4 sum_01_x4(float(0));
                                float32x4 sum_10_x4(float(0)); float32x4 sum_11_x4(float(0));
                                float32x4 sum_20_x4(float(0)); float32x4 sum_21_x4(float(0));
                                float32x4 sum_30_x4(float(0)); float32x4 sum_31_x4(float(0));

                                sum_00_x4 += r00 * k00;
                                sum_00_x4 += r01 * k01;
                                sum_00_x4 += r02 * k02;

                                sum_01_x4 += r01 * k00;
                                sum_01_x4 += r02 * k01;
                                sum_01_x4 += r03 * k02;

                                sum_10_x4 += r00 * k10;
                                sum_10_x4 += r01 * k11;
                                sum_10_x4 += r02 * k12;

                                sum_11_x4 += r01 * k10;
                                sum_11_x4 += r02 * k11;
                                sum_11_x4 += r03 * k12;

                                sum_20_x4 += r00 * k20;
                                sum_20_x4 += r01 * k21;
                                sum_20_x4 += r02 * k22;

                                sum_21_x4 += r01 * k20;
                                sum_21_x4 += r02 * k21;
                                sum_21_x4 += r03 * k22;

                                sum_30_x4 += r00 * k30;
                                sum_30_x4 += r01 * k31;
                                sum_30_x4 += r02 * k32;

                                sum_31_x4 += r01 * k30;
                                sum_31_x4 += r02 * k31;
                                sum_31_x4 += r03 * k32;

                                *out_ptr_00 += ts::sum(sum_00_x4, 3);
                                *out_ptr_01 += ts::sum(sum_01_x4, 3);
                                *out_ptr_10 += ts::sum(sum_10_x4, 3);
                                *out_ptr_11 += ts::sum(sum_11_x4, 3);
                                *out_ptr_20 += ts::sum(sum_20_x4, 3);
                                *out_ptr_21 += ts::sum(sum_21_x4, 3);
                                *out_ptr_30 += ts::sum(sum_30_x4, 3);
                                *out_ptr_31 += ts::sum(sum_31_x4, 3);

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_01++;
                                out_ptr_10++;
                                out_ptr_11++;
                                out_ptr_20++;
                                out_ptr_21++;
                                out_ptr_30++;
                                out_ptr_31++;

                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_00 += out_width;
                            out_ptr_01 += out_width;
                            out_ptr_10 += out_width;
                            out_ptr_11 += out_width;
                            out_ptr_20 += out_width;
                            out_ptr_21 += out_width;
                            out_ptr_30 += out_width;
                            out_ptr_31 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r01(r_1);
                                float32x4 r02(r_2);
                                float32x4 r03(r_3);

                                float32x4 sum_00(float(0));
                                float32x4 sum_10(float(0));
                                float32x4 sum_20(float(0));
                                float32x4 sum_30(float(0));

                                sum_00 += r00 * k00;
                                sum_00 += r01 * k01;
                                sum_00 += r02 * k02;

                                sum_10 += r00 * k10;
                                sum_10 += r01 * k11;
                                sum_10 += r02 * k12;

                                sum_20 += r00 * k20;
                                sum_20 += r01 * k21;
                                sum_20 += r02 * k22;

                                sum_30 += r00 * k30;
                                sum_30 += r01 * k31;
                                sum_30 += r02 * k32;

                                *out_ptr_00 += ts::sum(sum_00, 3);
                                *out_ptr_10 += ts::sum(sum_10, 3);
                                *out_ptr_20 += ts::sum(sum_20, 3);
                                *out_ptr_30 += ts::sum(sum_30, 3);

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_00++;
                                out_ptr_10++;
                                out_ptr_20++;
                                out_ptr_30++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        k_0 += 9;
                        k_1 += 9;
                        k_2 += 9;
                        k_3 += 9;
                    }
                }
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int p = remain_out_channel; p < out_channel; p++)
                {
                    float *out_at = poutput + n * out_num_offset + p * out_channel_offset;

                    const float* kernel_cur = pweight + p * input_channel * 9;

                    for (int q = 0; q < input_channel; q++)
                    {
                        float* out_ptr_0 = out_at;
                        float* out_ptr_1 = out_ptr_0 + out_width;

                        const float* input_cur = pinput + n * input_num_offset + q * input_channel_offset;

                        const float* r_0 = input_cur;
                        const float* r_1 = r_0 + input_width;
                        const float* r_2 = r_1 + input_width;
                        const float* r_3 = r_2 + input_width;

                        const float* k_0 = kernel_cur;
                        const float* k_1 = kernel_cur + 3;
                        const float* k_2 = kernel_cur + 6;

                        float32x4 k00(k_0);
                        float32x4 k10(k_1);
                        float32x4 k20(k_2);

                        int i = 0;
                        for (; i + 1 < out_height; i += 2)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);
                                float32x4 r30(r_3);

                                float32x4 sum00(float(0));
                                float32x4 sum10(float(0));

                                sum00 += r00 * k00;
                                sum00 += r10 * k10;
                                sum00 += r20 * k20;

                                sum10 += r10 * k00;
                                sum10 += r20 * k10;
                                sum10 += r30 * k20;

                                float sum_0 = ts::sum(sum00, 3);
                                float sum_1 = ts::sum(sum10, 3);

                                *out_ptr_0 += sum_0;
                                *out_ptr_1 += sum_1;

                                r_0++;
                                r_1++;
                                r_2++;
                                r_3++;
                                out_ptr_0++;
                                out_ptr_1++;
                            }

                            r_0 += 2 + input_width;
                            r_1 += 2 + input_width;
                            r_2 += 2 + input_width;
                            r_3 += 2 + input_width;

                            out_ptr_0 += out_width;
                            out_ptr_1 += out_width;
                        }

                        for (; i < out_height; i++)
                        {
                            int remain = out_width;
                            for (; remain > 0; remain--)
                            {
                                float32x4 r00(r_0);
                                float32x4 r10(r_1);
                                float32x4 r20(r_2);
                                float32x4 r30(r_3);

                                float32x4 sum00(float(0));

                                sum00 += r00 * k00;
                                sum00 += r00 * k10;
                                sum00 += r00 * k20;

                                float sum = ts::sum(sum00, 3);

                                *out_ptr_0 += sum;

                                r_0++;
                                r_1++;
                                r_2++;
                                out_ptr_0++;
                            }

                            r_0 += 2;
                            r_1 += 2;
                            r_2 += 2;
                        }

                        kernel_cur += 9;
                    }
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::kernel_pack4x4(const Tensor &kernel, Tensor& kernel_packed) {
            auto shape = kernel.sizes();
            int kernel_num = shape[0];
            int kernel_channel = shape[1];
            int kernel_h = shape[2];
            int kernel_w = shape[3];
            int num_offset = kernel_channel * kernel_h * kernel_w;
            const T* pkernel = kernel.data<T>();
            T* pkernel_packed = kernel_packed.data<T>();

            int out_loop = kernel_num >> 2;
            int remain = out_loop << 2;

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++){
                int n = nn * 4;

                const T* k0 = pkernel + n * num_offset;
                const T* k1 = k0 + num_offset;
                const T* k2 = k1 + num_offset;
                const T* k3 = k2 + num_offset;

                T* kernel_packed_at = pkernel_packed + n * num_offset;

                for (int i = 0; i < num_offset; i++) {
                    *kernel_packed_at++ = *k0++;
                    *kernel_packed_at++ = *k1++;
                    *kernel_packed_at++ = *k2++;
                    *kernel_packed_at++ = *k3++;
                }
            }

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < kernel_num; n++){
                const T* k0 = pkernel + n * num_offset;
                T* kernel_packed_at = pkernel_packed + n * num_offset;
                for (int i = 0; i < num_offset; i++) {
                    *kernel_packed_at++ = *k0++;
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::col_pack4x4(const T* col_tensor, int col_h, int col_w, T* col_packed) {
            const T* pcol = col_tensor;
            T* pcol_packed = col_packed;

            int out_loop = col_w >> 2;
            int remain = out_loop << 2;

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++){
                int n = nn * 4;
                const T* col_at = pcol + n;
                T* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++){
                    *packed_at++ = col_at[0];
                    *packed_at++ = col_at[1];
                    *packed_at++ = col_at[2];
                    *packed_at++ = col_at[3];

                    col_at += col_w;
                }
            }
#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < col_w; n++){
                const T* col_at = pcol + n;
                T* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++) {
                    *packed_at++ = col_at[0];
                    col_at += col_w;
                }
            }
        }

        template<>
        void Conv2dAlgorithm<float>::col_pack4x4(const float* col_tensor, int col_h, int col_w, float* col_packed) {
            const float* pcol = col_tensor;
            float* pcol_packed = col_packed;

            int out_loop = col_w >> 2;
            int remain = out_loop << 2;

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++) {
                int n = nn * 4;
                const float* col_at = pcol + n;
                float* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++) {
                    float32x4 col_at_x4(col_at);
                    col_at_x4.store(packed_at);

                    col_at += col_w;
                    packed_at += 4;
                }
            }
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < col_w; n++) {
                const float* col_at = pcol + n;
                float* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++) {
                    *packed_at++ = col_at[0];
                    col_at += col_w;
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::gemm_pack4x4(int M, int N, int K, const T* kernel_packed, const T* col_packed, T* out) {

        }

        template<>
        void Conv2dAlgorithm<float>::gemm_pack4x4(int M, int N, int K, const float* kernel_packed, const float* col_packed, float* out) {
            const float* pkernel_packed = kernel_packed;
            const float* pcol_packed = col_packed;
            float* pout = out;

            int out_channel_offset = N;
            int kernel_num_offset = K;

            int out_loop = M >> 2;
            int remain = out_loop << 2;
            float* output_at = pout;
#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int mm = 0; mm < out_loop; mm++){
                int m = mm * 4;
                float* output_row0 = output_at + m * out_channel_offset;
                float* output_row1 = output_row0 + out_channel_offset;
                float* output_row2 = output_row1 + out_channel_offset;
                float* output_row3 = output_row2 + out_channel_offset;

                const float* kernel_store = pkernel_packed + m * kernel_num_offset;

                int n_loop = N >> 2;
                int n_remain = n_loop << 2;
                for (int nn = 0; nn < n_loop; nn++){
                    int n = nn * 4;
                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;

                    float32x4 c0(0.f), c1(0.f), c2(0.f), c3(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){
                        //=====================pack_gemm k==0=====================
                        float32x4 k0 = broadcast2float32x4(kernel_at);       //[k00,k00,k00,k00]
                        float32x4 k1 = broadcast2float32x4(kernel_at + 1);   //[k10,k10,k10,k10]
                        float32x4 k2 = broadcast2float32x4(kernel_at + 2);   //[k20,k20,k20,k20]
                        float32x4 k3 = broadcast2float32x4(kernel_at + 3);   //[k30,k30,k30,k30]

                        float32x4 a0(col_at);                                //[a00,a01,a02,a03]

                        c0 = fmadd(a0, k0, c0);
                        c1 = fmadd(a0, k1, c1);
                        c2 = fmadd(a0, k2, c2);
                        c3 = fmadd(a0, k3, c3);

                        //=====================pack_gemm k==1=====================
                        k0 = broadcast2float32x4(kernel_at + 4);              //[k01,k01,k01,k01]
                        k1 = broadcast2float32x4(kernel_at + 5);              //[k11,k11,k11,k11]
                        k2 = broadcast2float32x4(kernel_at + 6);              //[k21,k21,k21,k21]
                        k3 = broadcast2float32x4(kernel_at + 7);              //[k31,k31,k31,k31]

                        float32x4 a1(col_at + 4);                             //[a10,a11,a12,a13]

                        c0 = fmadd(a1, k0, c0);
                        c1 = fmadd(a1, k1, c1);
                        c2 = fmadd(a1, k2, c2);
                        c3 = fmadd(a1, k3, c3);

                        //=====================pack_gemm k==2=====================
                        k0 = broadcast2float32x4(kernel_at + 8);              //[k02,k02,k02,k02]
                        k1 = broadcast2float32x4(kernel_at + 9);              //[k12,k12,k12,k12]
                        k2 = broadcast2float32x4(kernel_at + 10);             //[k22,k21,k21,k21]
                        k3 = broadcast2float32x4(kernel_at + 11);             //[k32,k32,k32,k32]

                        float32x4 a2(col_at + 8);                             //[a20,a21,a22,a23]

                        c0 = fmadd(a2, k0, c0);
                        c1 = fmadd(a2, k1, c1);
                        c2 = fmadd(a2, k2, c2);
                        c3 = fmadd(a2, k3, c3);

                        //=====================pack_gemm k==3=====================
                        k0 = broadcast2float32x4(kernel_at + 12);              //[k03,k03,k03,k03]
                        k1 = broadcast2float32x4(kernel_at + 13);              //[k13,k13,k13,k13]
                        k2 = broadcast2float32x4(kernel_at + 14);              //[k23,k23,k23,k23]
                        k3 = broadcast2float32x4(kernel_at + 15);              //[k33,k33,k33,k33]

                        float32x4 a3(col_at + 12);                             //[a30,a31,a32,a33]

                        c0 = fmadd(a3, k0, c0);
                        c1 = fmadd(a3, k1, c1);
                        c2 = fmadd(a3, k2, c2);
                        c3 = fmadd(a3, k3, c3);

                        kernel_at += 16;
                        col_at += 16;
                    }

                    for (int k = k_remain; k < K; k++){
                        float32x4 k0 = broadcast2float32x4(kernel_at);       //[k00,k00,k00,k00]
                        float32x4 k1 = broadcast2float32x4(kernel_at + 1);   //[k10,k10,k10,k10]
                        float32x4 k2 = broadcast2float32x4(kernel_at + 2);   //[k20,k20,k20,k20]
                        float32x4 k3 = broadcast2float32x4(kernel_at + 3);   //[k30,k30,k30,k30]

                        float32x4 a0(col_at);                                //[a00,a01,a02,a03]

                        c0 = fmadd(a0, k0, c0);
                        c1 = fmadd(a0, k1, c1);
                        c2 = fmadd(a0, k2, c2);
                        c3 = fmadd(a0, k3, c3);

                        kernel_at += 4;
                        col_at += 4;
                    }

                    c0.store(output_row0); c1.store(output_row1);
                    c2.store(output_row2); c3.store(output_row3);

                    output_row0 += 4; output_row1 += 4;
                    output_row2 += 4; output_row3 += 4;
                }

                for (int n = n_remain; n < N; n++){
                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;
                    float32x4 sum_col0(0.f), sum_col1(0.f), sum_col2(0.f), sum_col3(0.f);
                    float32x4 sum_col(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){
                        // int k = kk * 4;
                        float32x4 a0 = broadcast2float32x4(col_at);          //[a00,a00,a00,a00]
                        float32x4 a1 = broadcast2float32x4(col_at + 1);      //[a10,a10,a10,a10]
                        float32x4 a2 = broadcast2float32x4(col_at + 2);      //[a20,a20,a20,a20]
                        float32x4 a3 = broadcast2float32x4(col_at + 3);      //[a30,a30,a30,a30]

                        float32x4 k0(kernel_at);                             //[k00,k10,k20,k30]
                        float32x4 k1(kernel_at + 4);                         //[k01,k11,k21,k31]
                        float32x4 k2(kernel_at + 8);                         //[k02,k12,k22,k32]
                        float32x4 k3(kernel_at + 12);                        //[k03,k13,k23,k33]

                        sum_col0 = fmadd(k0, a0, sum_col0);
                        sum_col1 = fmadd(k1, a1, sum_col1);
                        sum_col2 = fmadd(k2, a2, sum_col2);
                        sum_col3 = fmadd(k3, a3, sum_col3);

                        kernel_at += 16;
                        col_at += 4;
                    }

                    sum_col0 += sum_col1;
                    sum_col2 += sum_col3;
                    sum_col += sum_col0;
                    sum_col += sum_col2;

                    for (int k = k_remain; k < K; k++){
                        float32x4 a0 = broadcast2float32x4(col_at);          //[a00,a00,a00,a00]
                        float32x4 k0(kernel_at);                             //[k00,k10,k20,k30]

                        sum_col = fmadd(k0, a0, sum_col);

                        kernel_at += 4;
                        col_at += 1;
                    }

                    *output_row0++ = *((float*)&sum_col.value);
                    *output_row1++ = *(((float*)&sum_col.value) + 1);
                    *output_row2++ = *(((float*)&sum_col.value) + 2);
                    *output_row3++ = *(((float*)&sum_col.value) + 3);
                }
            }

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = remain; m < M; m++){
                float* output_row0 = output_at + m * out_channel_offset;
                const float* kernel_store = pkernel_packed + m * kernel_num_offset;

                int n_loop = N >> 2;
                int n_remain = n_loop << 2;
                for (int nn = 0; nn < n_loop; nn++){
                    int n = nn * 4;
                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;

                    float32x4 c0(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){
                        float32x4 k0 = broadcast2float32x4(kernel_at);       //[k00,k00,k00,k00]
                        float32x4 k1 = broadcast2float32x4(kernel_at + 1);   //[k01,k01,k01,k01]
                        float32x4 k2 = broadcast2float32x4(kernel_at + 2);   //[k02,k02,k02,k02]
                        float32x4 k3 = broadcast2float32x4(kernel_at + 3);   //[k03,k03,k03,k03]

                        float32x4 a0(col_at);                                //[a00,a01,a02,a03]
                        float32x4 a1(col_at + 4);                            //[a10,a11,a12,a13]
                        float32x4 a2(col_at + 8);                            //[a20,a21,a22,a23]
                        float32x4 a3(col_at + 12);                           //[a30,a31,a32,a33]

                        c0 = fmadd(k0, a0, c0);
                        c0 = fmadd(k1, a1, c0);
                        c0 = fmadd(k2, a2, c0);
                        c0 = fmadd(k3, a3, c0);

                        kernel_at += 4;
                        col_at += 16;
                    }

                    for (int k = k_remain; k < K; k++){
                        float32x4 k0 = broadcast2float32x4(kernel_at);        //[k00,k00,k00,k00]
                        float32x4 a0(col_at);                                 //[a00,a01,a02,a03]

                        c0 = fmadd(k0, a0, c0);

                        kernel_at += 1;
                        col_at += 4;
                    }

                    c0.store(output_row0);
                    output_row0 += 4;
                }

                for (int n = n_remain; n < N; n++){
                    float32x4 c0(0.f);
                    float sum0 = 0;

                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){
                        // int k = kk * 4;
                        float32x4 k0(kernel_at);
                        float32x4 a0(col_at);

                        c0 = fmadd(k0, a0, c0);

                        kernel_at += 4;
                        col_at += 4;
                    }

                    sum0 = ts::sum(c0);

                    for (int k = k_remain; k < K; k++) {
                        sum0 += (*kernel_at) * (*col_at);
                        kernel_at++;
                        col_at++;
                    }

                    *output_row0 = sum0;
                    output_row0++;
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::kernel_pack8x8(const Tensor &kernel, Tensor& kernel_packed) {
            auto shape = kernel.sizes();
            int kernel_num = shape[0];
            int kernel_channel = shape[1];
            int kernel_h = shape[2];
            int kernel_w = shape[3];
            int num_offset = kernel_channel * kernel_h * kernel_w;
            const T* pkernel = kernel.data<T>();
            T* pkernel_packed = kernel_packed.data<T>();

            int out_loop = shape[0] >> 3;
            int remain = out_loop << 3;

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++){
                int n = nn * 8;
                const T* k0 = pkernel + n * num_offset;
                const T* k1 = k0 + num_offset;
                const T* k2 = k1 + num_offset;
                const T* k3 = k2 + num_offset;
                const T* k4 = k3 + num_offset;
                const T* k5 = k4 + num_offset;
                const T* k6 = k5 + num_offset;
                const T* k7 = k6 + num_offset;

                T* kernel_packed_at = pkernel_packed + n * num_offset;

                for (int i = 0; i < num_offset; i++){
                    *kernel_packed_at++ = *k0++;
                    *kernel_packed_at++ = *k1++;
                    *kernel_packed_at++ = *k2++;
                    *kernel_packed_at++ = *k3++;
                    *kernel_packed_at++ = *k4++;
                    *kernel_packed_at++ = *k5++;
                    *kernel_packed_at++ = *k6++;
                    *kernel_packed_at++ = *k7++;
                }
            }
            //NOTE:Maybe i should pack 4x4 on remain size
#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < kernel_num; n++){
                const T* k0 = pkernel + n * num_offset;
                T* kernel_packed_at = pkernel_packed + n * num_offset;
                for (int i = 0; i < num_offset; i++) {
                    *kernel_packed_at++ = *k0++;
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::col_pack8x8(const T* col_tensor, int col_h, int col_w, T* col_packed) {
            const T* pcol = col_tensor;
            T* pcol_packed = col_packed;

            int out_loop = col_w >> 3;
            int remain = out_loop << 3;

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++) {
                int n = nn * 8;
                const T* col_at = pcol + n;
                T* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++){
                    *packed_at++ = col_at[0];
                    *packed_at++ = col_at[1];
                    *packed_at++ = col_at[2];
                    *packed_at++ = col_at[3];
                    *packed_at++ = col_at[4];
                    *packed_at++ = col_at[5];
                    *packed_at++ = col_at[6];
                    *packed_at++ = col_at[7];

                    col_at += col_w;
                }
            }
#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < col_w; n++){
                const T* col_at = pcol + n;
                T* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++) {
                    *packed_at++ = col_at[0];
                    col_at += col_w;
                }
            }
        }

        template<>
        void Conv2dAlgorithm<float>::col_pack8x8(const float* col_tensor, int col_h, int col_w, float* col_packed) {
            const float* pcol = col_tensor;
            float* pcol_packed = col_packed;

            int out_loop = col_w >> 3;
            int remain = out_loop << 3;

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++) {
                int n = nn * 8;
                const float* col_at = pcol + n;
                float* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++) {
                    float32x4x2 col_at_x4x2(col_at);
                    col_at_x4x2.store(packed_at);
                    col_at += col_w;
                    packed_at += 8;
                }
            }
#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < col_w; n++) {
                const float* col_at = pcol + n;
                float* packed_at = pcol_packed + n * col_h;

                for (int i = 0; i < col_h; i++) {
                    *packed_at++ = col_at[0];
                    col_at += col_w;
                }
            }
        }

        template<typename T>
        void Conv2dAlgorithm<T>::gemm_pack8x8(int M, int N, int K, const T* kernel_packed, const T* col_packed, T* out) {
        
        }

        template<>
        void Conv2dAlgorithm<float>::gemm_pack8x8(int M, int N, int K, const float* kernel_packed, const float* col_packed, float* out) {
            
            const float* pkernel_packed = kernel_packed;
            const float* pcol_packed = col_packed;
            float* pout = out;

            //auto out_shape = out.sizes();
            //int out_channel_offset = out_shape[2] * out_shape[3];
            int out_channel_offset = N;
            int kernel_num_offset = K;

            int out_loop = M >> 3;
            int remain = out_loop << 3;
            float* output_at = pout;
#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int mm = 0; mm < out_loop; mm++){
                int m = mm * 8;
                float* output_row0 = output_at + m * out_channel_offset;
                float* output_row1 = output_row0 + out_channel_offset;
                float* output_row2 = output_row1 + out_channel_offset;
                float* output_row3 = output_row2 + out_channel_offset;
                float* output_row4 = output_row3 + out_channel_offset;
                float* output_row5 = output_row4 + out_channel_offset;
                float* output_row6 = output_row5 + out_channel_offset;
                float* output_row7 = output_row6 + out_channel_offset;

                const float* kernel_store = pkernel_packed + m * kernel_num_offset;

                int n_loop = N >> 3;
                int n_remain = n_loop << 3;
                for (int nn = 0; nn < n_loop; nn++)
                {
                    int n = nn * 8;
                    
                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;

                    float32x4x2 c0(0.f), c1(0.f), c2(0.f), c3(0.f);
                    float32x4x2 c4(0.f), c5(0.f), c6(0.f), c7(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){
                        //=====================pack_gemm k==0=====================
                        float32x4x2 k0 = broadcast2float32x4x2(kernel_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 k1 = broadcast2float32x4x2(kernel_at + 1);   //[k10,k10,k10,k10,k10,k10,k10,k10]
                        float32x4x2 k2 = broadcast2float32x4x2(kernel_at + 2);   //[k20,k20,k20,k20,k20,k20,k20,k20]
                        float32x4x2 k3 = broadcast2float32x4x2(kernel_at + 3);   //[k30,k30,k30,k30,k30,k30,k30,k30]

                        float32x4x2 a0(col_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]
                        
                        c0 = fmadd(a0, k0, c0);
                        c1 = fmadd(a0, k1, c1);
                        c2 = fmadd(a0, k2, c2);
                        c3 = fmadd(a0, k3, c3);
                        //Note:The number of registers is limited
                        k0 = broadcast2float32x4x2(kernel_at + 4);               //[k40,k40,k40,k40,k40,k40,k40,k40]
                        k1 = broadcast2float32x4x2(kernel_at + 5);               //[k50,k50,k50,k50,k50,k50,k50,k50]
                        k2 = broadcast2float32x4x2(kernel_at + 6);               //[k60,k60,k60,k60,k60,k60,k60,k60]
                        k3 = broadcast2float32x4x2(kernel_at + 7);               //[k70,k70,k70,k70,k70,k70,k70,k70]

                        c4 = fmadd(a0, k0, c4);
                        c5 = fmadd(a0, k1, c5);
                        c6 = fmadd(a0, k2, c6);
                        c7 = fmadd(a0, k3, c7);

                        //=====================pack_gemm k==1=====================
                        k0 = broadcast2float32x4x2(kernel_at + 8);               //[k01,k01,k01,k01,k01,k01,k01,k01]
                        k1 = broadcast2float32x4x2(kernel_at + 9);               //[k11,k11,k11,k11,k11,k11,k11,k11]
                        k2 = broadcast2float32x4x2(kernel_at + 10);              //[k21,k21,k21,k21,k21,k21,k21,k21]
                        k3 = broadcast2float32x4x2(kernel_at + 11);              //[k31,k31,k31,k31,k31,k31,k31,k31]

                        float32x4x2 a1(col_at + 8);                              //[a10,a11,a12,a13,a14,a15,a16,a17]

                        c0 = fmadd(a1, k0, c0);
                        c1 = fmadd(a1, k1, c1);
                        c2 = fmadd(a1, k2, c2);
                        c3 = fmadd(a1, k3, c3);

                        k0 = broadcast2float32x4x2(kernel_at + 12);              //[k41,k41,k41,k41,k41,k41,k41,k41]
                        k1 = broadcast2float32x4x2(kernel_at + 13);              //[k51,k51,k51,k51,k51,k51,k51,k51]
                        k2 = broadcast2float32x4x2(kernel_at + 14);              //[k61,k61,k61,k61,k61,k61,k61,k61]
                        k3 = broadcast2float32x4x2(kernel_at + 15);              //[k71,k71,k71,k71,k71,k71,k71,k71]

                        c4 = fmadd(a1, k0, c4);
                        c5 = fmadd(a1, k1, c5);
                        c6 = fmadd(a1, k2, c6);
                        c7 = fmadd(a1, k3, c7);
                        //=====================pack_gemm k==2=====================
                        k0 = broadcast2float32x4x2(kernel_at + 16);              //[k02,k02,k02,k02,k02,k02,k02,k02]
                        k1 = broadcast2float32x4x2(kernel_at + 17);              //[k12,k12,k12,k12,k12,k12,k12,k12]
                        k2 = broadcast2float32x4x2(kernel_at + 18);              //[k22,k21,k21,k21,k21,k21,k21,k21]
                        k3 = broadcast2float32x4x2(kernel_at + 19);              //[k32,k32,k32,k32,k32,k32,k32,k32]

                        float32x4x2 a2(col_at + 16);                             //[a20,a21,a22,a23,a24,a25,a26,a27]

                        c0 = fmadd(a2, k0, c0);
                        c1 = fmadd(a2, k1, c1);
                        c2 = fmadd(a2, k2, c2);
                        c3 = fmadd(a2, k3, c3);

                        k0 = broadcast2float32x4x2(kernel_at + 20);              //[k42,k42,k42,k42,k42,k42,k42,k42]
                        k1 = broadcast2float32x4x2(kernel_at + 21);              //[k52,k52,k52,k52,k52,k52,k52,k52]
                        k2 = broadcast2float32x4x2(kernel_at + 22);              //[k62,k62,k62,k62,k62,k62,k62,k62]
                        k3 = broadcast2float32x4x2(kernel_at + 23);              //[k72,k72,k72,k72,k72,k72,k72,k72]

                        c4 = fmadd(a2, k0, c4);
                        c5 = fmadd(a2, k1, c5);
                        c6 = fmadd(a2, k2, c6);
                        c7 = fmadd(a2, k3, c7);
                        //=====================pack_gemm k==3=====================
                        k0 = broadcast2float32x4x2(kernel_at + 24);              //[k03,k03,k03,k03,k03,k03,k03,k03]
                        k1 = broadcast2float32x4x2(kernel_at + 25);              //[k13,k13,k13,k13,k13,k13,k13,k13]
                        k2 = broadcast2float32x4x2(kernel_at + 26);              //[k23,k23,k23,k23,k23,k23,k23,k23]
                        k3 = broadcast2float32x4x2(kernel_at + 27);              //[k33,k33,k33,k33,k33,k33,k33,k33]

                        float32x4x2 a3(col_at + 24);                             //[a30,a31,a32,a33,a34,a35,a36,a37]

                        c0 = fmadd(a3, k0, c0);
                        c1 = fmadd(a3, k1, c1);
                        c2 = fmadd(a3, k2, c2);
                        c3 = fmadd(a3, k3, c3);

                        k0 = broadcast2float32x4x2(kernel_at + 28);              //[k43,k43,k43,k43,k43,k43,k43,k43]
                        k1 = broadcast2float32x4x2(kernel_at + 29);              //[k53,k53,k53,k53,k53,k53,k53,k53]
                        k2 = broadcast2float32x4x2(kernel_at + 30);              //[k63,k63,k63,k63,k63,k63,k63,k63]
                        k3 = broadcast2float32x4x2(kernel_at + 31);              //[k73,k73,k73,k73,k73,k73,k73,k73]

                        c4 = fmadd(a3, k0, c4);
                        c5 = fmadd(a3, k1, c5);
                        c6 = fmadd(a3, k2, c6);
                        c7 = fmadd(a3, k3, c7);

                        kernel_at += 32;
                        col_at += 32;
                    }

                    for (int k = k_remain; k < K; k++){
                        float32x4x2 k0 = broadcast2float32x4x2(kernel_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 k1 = broadcast2float32x4x2(kernel_at + 1);   //[k10,k10,k10,k10,k10,k10,k10,k10]
                        float32x4x2 k2 = broadcast2float32x4x2(kernel_at + 2);   //[k20,k20,k20,k20,k20,k20,k20,k20]
                        float32x4x2 k3 = broadcast2float32x4x2(kernel_at + 3);   //[k30,k30,k30,k30,k30,k30,k30,k30]

                        float32x4x2 a0(col_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]

                        c0 = fmadd(a0, k0, c0);
                        c1 = fmadd(a0, k1, c1);
                        c2 = fmadd(a0, k2, c2);
                        c3 = fmadd(a0, k3, c3);

                        k0 = broadcast2float32x4x2(kernel_at + 4);               //[k40,k40,k40,k40,k40,k40,k40,k40]
                        k1 = broadcast2float32x4x2(kernel_at + 5);               //[k50,k50,k50,k50,k50,k50,k50,k50]
                        k2 = broadcast2float32x4x2(kernel_at + 6);               //[k60,k60,k60,k60,k60,k60,k60,k60]
                        k3 = broadcast2float32x4x2(kernel_at + 7);               //[k70,k70,k70,k70,k70,k70,k70,k70]

                        c4 = fmadd(a0, k0, c4);
                        c5 = fmadd(a0, k1, c5);
                        c6 = fmadd(a0, k2, c6);
                        c7 = fmadd(a0, k3, c7);

                        kernel_at += 8;
                        col_at += 8;
                    }

                    c0.store(output_row0); c1.store(output_row1);
                    c2.store(output_row2); c3.store(output_row3);
                    c4.store(output_row4); c5.store(output_row5);
                    c6.store(output_row6); c7.store(output_row7);

                    output_row0 += 8;output_row1 += 8;
                    output_row2 += 8;output_row3 += 8;
                    output_row4 += 8;output_row5 += 8;
                    output_row6 += 8;output_row7 += 8;
                }
           
                for (int n = n_remain; n < N; n++)
                {
                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;
                    float32x4x2 sum_col0(0.f), sum_col1(0.f), sum_col2(0.f), sum_col3(0.f);
                    float32x4x2 sum_col(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){
                        // int k = kk * 4;
                        
                        float32x4x2 a0 = broadcast2float32x4x2(col_at);          //[a00,a00,a00,a00,a00,a00,a00,a00]
                        float32x4x2 a1 = broadcast2float32x4x2(col_at + 1);      //[a10,a10,a10,a10,a10,a10,a10,a10]
                        float32x4x2 a2 = broadcast2float32x4x2(col_at + 2);      //[a20,a20,a20,a20,a20,a20,a20,a20]
                        float32x4x2 a3 = broadcast2float32x4x2(col_at + 3);      //[a30,a30,a30,a30,a30,a30,a30,a30]

                        float32x4x2 k0(kernel_at);                               //[k00,k10,k20,k30,k40,k50,k60,k70]
                        float32x4x2 k1(kernel_at + 8);                           //[k01,k11,k21,k31,k41,k51,k61,k71]
                        float32x4x2 k2(kernel_at + 16);                          //[k02,k12,k22,k32,k42,k52,k62,k72]
                        float32x4x2 k3(kernel_at + 24);                          //[k03,k13,k23,k33,k43,k53,k63,k73]

                        sum_col0 = fmadd(k0, a0, sum_col0);
                        sum_col1 = fmadd(k1, a1, sum_col1);
                        sum_col2 = fmadd(k2, a2, sum_col2);
                        sum_col3 = fmadd(k3, a3, sum_col3);

                        kernel_at += 32;
                        col_at += 4;
                    }

                    sum_col0 += sum_col1;
                    sum_col2 += sum_col3;
                    sum_col += sum_col0;
                    sum_col += sum_col2;

                    for (int k = k_remain; k < K; k++){
                        float32x4x2 a0 = broadcast2float32x4x2(col_at);          //[a00,a00,a00,a00,a00,a00,a00,a00]
                        float32x4x2 k0(kernel_at);                               //[k00,k10,k20,k30,k40,k50,k60,k70]

                        sum_col = fmadd(k0, a0, sum_col);

                        kernel_at += 8;
                        col_at += 1;
                    }

                    *output_row0++ = *((float*)&sum_col.value);
                    *output_row1++ = *(((float*)&sum_col.value) + 1);
                    *output_row2++ = *(((float*)&sum_col.value) + 2);
                    *output_row3++ = *(((float*)&sum_col.value) + 3);
                    *output_row4++ = *(((float*)&sum_col.value) + 4);
                    *output_row5++ = *(((float*)&sum_col.value) + 5);
                    *output_row6++ = *(((float*)&sum_col.value) + 6);
                    *output_row7++ = *(((float*)&sum_col.value) + 7);
                }
            }

#ifdef TS_USE_OPENMP
            #pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int m = remain; m < M; m++){
                float* output_row0 = output_at + m * out_channel_offset;
                const float* kernel_store = pkernel_packed + m * kernel_num_offset;

                int n_loop = N >> 3;
                int n_remain = n_loop << 3;
                for (int nn = 0; nn < n_loop; nn++){
                    int n = nn * 8;

                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;

                    float32x4x2 c0(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){

                        float32x4x2 k0 = broadcast2float32x4x2(kernel_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 k1 = broadcast2float32x4x2(kernel_at + 1);   //[k01,k01,k01,k01,k01,k01,k01,k01]
                        float32x4x2 k2 = broadcast2float32x4x2(kernel_at + 2);   //[k02,k02,k02,k02,k02,k02,k02,k02]
                        float32x4x2 k3 = broadcast2float32x4x2(kernel_at + 3);   //[k03,k03,k03,k03,k03,k03,k03,k03]

                        float32x4x2 a0(col_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]
                        float32x4x2 a1(col_at + 8);                              //[a10,a11,a12,a13,a14,a15,a16,a17]
                        float32x4x2 a2(col_at + 16);                             //[a20,a21,a22,a23,a24,a25,a26,a27]
                        float32x4x2 a3(col_at + 24);                             //[a30,a31,a32,a33,a34,a35,a36,a37]

                        c0 = fmadd(k0, a0, c0);
                        c0 = fmadd(k1, a1, c0);
                        c0 = fmadd(k2, a2, c0);
                        c0 = fmadd(k3, a3, c0);

                        kernel_at += 4;
                        col_at += 32;
                    }

                    for (int k = k_remain; k < K; k++){
                        float32x4x2 k0 = broadcast2float32x4x2(kernel_at);        //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 a0(col_at);                                   //[a00,a01,a02,a03,a04,a05,a06,a07]

                        c0 = fmadd(k0, a0, c0);

                        kernel_at += 1;
                        col_at += 8;
                    }

                    c0.store(output_row0);
                    output_row0 += 8;
                }

                for (int n = n_remain; n < N; n++){
                    float32x4 c0(0.f);
                    float sum0 = 0;

                    const float* kernel_at = kernel_store;
                    const float* col_at = pcol_packed + n * kernel_num_offset;

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++){
                        // int k = kk * 4;
                        float32x4 k0(kernel_at);
                        float32x4 a0(col_at);

                        c0 = fmadd(k0, a0, c0);

                        kernel_at += 4;
                        col_at += 4;
                    }

                    sum0 = ts::sum(c0);

                    for (int k = k_remain; k < K; k++){
                        sum0 += (*kernel_at) * (*col_at);
                        kernel_at++;
                        col_at++;
                    }

                    *output_row0 = sum0;
                    output_row0++;
                }
            }
        }
    }
}

template class ts::cpu::Conv2dAlgorithm<ts::dtype<ts::FLOAT32>::declare>;
template class ts::cpu::Conv2dAlgorithm<ts::dtype<ts::FLOAT64>::declare>;