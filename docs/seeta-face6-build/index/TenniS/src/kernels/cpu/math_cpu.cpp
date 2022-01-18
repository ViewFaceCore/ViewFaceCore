//
// Created by kier on 2018/7/19.
//

#include "kernels/cpu/math_cpu.h"
#include "kernels/common/math.h"
#include "utils/assert.h"
#include "runtime/inside/thread_pool.h"
#include "utils/ctxmgr.h"
#include "utils/box.h"

#include <iostream>

#include <cmath>

#include <runtime/inside/parallel.h>

#include "kernels/common/openmp.h"
#include "kernels/common/simd.h"


#include <core/dtype.h>
#include <core/tensor.h>


namespace ts {
    namespace cpu {

        template<typename T_IN,typename T_OUT>
        inline T_OUT inline_dot(int N, const T_IN *x, int incx, const T_IN *y, int incy) {
            T_OUT sum = 0;

            const int BLOCK = 4;
            int BODY = N / BLOCK, TAIL = N % BLOCK;

            for (; BODY; --BODY) {
                sum += *x * *y; x += incx; y += incy;
                sum += *x * *y; x += incx; y += incy;
                sum += *x * *y; x += incx; y += incy;
                sum += *x * *y; x += incx; y += incy;
            }
            for (; TAIL; --TAIL) {
                sum += *x * *y; x += incx; y += incy;
            }
            return sum;
        }
#ifdef TS_USE_SIMD
        template <>
        inline float inline_dot<float,float>(int N, const float *x, int incx, const float *y, int incy) {
            const auto incx1 = incx;
            const auto incx2 = incx1 + incx;
            const auto incx3 = incx2 + incx;
            const auto incx4 = incx3 + incx;
            const auto incy1 = incy;
            const auto incy2 = incy1 + incy;
            const auto incy3 = incy2 + incy;
            const auto incy4 = incy3 + incy;

            float sum = 0;
            int i = 0;

            float32x4 sumx4 = 0;
            for (; i < N - 3; i += 4) {
                sumx4 += float32x4(x[0], x[incx1], x[incx2], x[incx3]) * float32x4(y[0], y[incy1], y[incy2], y[incy3]);
                x += incx4;
                y += incy4;
            }

            sum = ts::sum(sumx4);

            for (; i < N; ++i) {
                sum += *x * *y;
                x += incx;
                y += incy;
            }

            return sum;
        }
#endif

        template<typename T_IN>
        inline void inline_zero(int N, T_IN *x, int incx) {
            if (incx == 1) {
                std::memset(x, 0, N * sizeof(T_IN));
                return;
            }
            TS_PARALLEL_RANGE_BEGIN(range, 0, N)
                    auto xx = x + range.first * incx;
                    const auto count = range.second - range.first;
                    int i = 0;
                    for (; i < count - 3; i += 4) {
                        *xx = 0; xx += incx;
                        *xx = 0; xx += incx;
                        *xx = 0; xx += incx;
                        *xx = 0; xx += incx;
                    }
                    for (; i < count; ++i) {
                        *xx = 0; xx += incx;
                    }
            TS_PARALLEL_RANGE_END()
        }

        template<typename T_IN>
        inline void inline_scal(int N, T_IN alpha, T_IN *x, int incx) {
            if (ts::near(alpha, T_IN(1))) return; // TODO: update float number equal check method
            if (ts::near(alpha, T_IN(0))) {
                inline_zero<T_IN>(N, x, incx);
                return;
            }
            // use thread
            TS_PARALLEL_RANGE_BEGIN(range, 0, N)
                    auto xx = x + range.first * incx;
                    const auto count = range.second - range.first;
                    int i = 0;
                    for (; i < count - 3; i += 4) {
                        *xx *= alpha; xx += incx;
                        *xx *= alpha; xx += incx;
                        *xx *= alpha; xx += incx;
                        *xx *= alpha; xx += incx;
                    }
                    for (; i < count; ++i) {
                        *xx *= alpha; xx += incx;
                    }
            TS_PARALLEL_RANGE_END()
        }


        template<typename T_IN,typename T_OUT>
        T_OUT math<T_IN,T_OUT>::dot(int N, const T_IN *x, int incx, const T_IN *y, int incy) {
            std::vector<T_OUT> parallel_sum(TS_PARALLEL_SIZE, T_OUT(0));
            TS_PARALLEL_RANGE_BEGIN(range, 0, N)
                    auto xx = x + range.first * incx;
                    auto yy = y + range.first * incy;
                    const auto count = range.second - range.first;
                    parallel_sum[__parallel_id] += inline_dot<T_IN,T_OUT>(count, xx, incx, yy, incy);
            TS_PARALLEL_RANGE_END()
            T_OUT sum = 0;
            for (auto value : parallel_sum) sum += value;
            return sum;
        }

        template<typename T_IN,typename T_OUT>
        inline void inline_gemm_row_major(
                blas::Transpose TransA,
                blas::Transpose TransB,
                int M, int N, int K,
                T_IN alpha,
                const T_IN *A, int lda,
                const T_IN *B, int ldb,
                T_IN beta,
                T_OUT *C, int ldc) {
            // TODO: check if lda, ldb, ldc use correct
            TS_AUTO_CHECK(lda >= (TransA == blas::NoTrans ? K : M));
            TS_AUTO_CHECK(ldb >= (TransB == blas::NoTrans ? N : K));
            TS_AUTO_CHECK(ldc >= N);

            //auto gun = try_threads_on(size_t(M), 4);

            // calculate beta * C
            // C is RowMajor
            if (ldc == N) inline_scal<T_OUT>(M * N, beta, C, 1);
            else {
                TS_PARALLEL_FOR_BEGIN(m, 0, M)
                            auto CC = &C[m * ldc];
                            inline_scal<T_OUT>(N, beta, CC, 1);
                TS_PARALLEL_FOR_END()
            }

            if (ts::near(alpha, T_IN(0))) return;

            unsigned int condition = (TransA == blas::NoTrans ? 0U : 1U) | ((TransB == blas::NoTrans ? 0U : 2U));
            switch (condition) {
                case 0: // A: NoTrans, B: NoTrans
                TS_PARALLEL_FOR_BEGIN(i, 0, M)
                            T_OUT *C_anchor = &C[i * ldc];
                            for (int j = 0; j < N; ++j) {
                                *C_anchor += alpha * inline_dot<T_IN, T_OUT>(K, &A[i * lda], 1, &B[j], ldb);
                                C_anchor++;
                            }
                TS_PARALLEL_FOR_END()
                    break;
                case 1: // A: Trans, B: NoTrans
                TS_PARALLEL_FOR_BEGIN(i, 0, M)
                            T_OUT *C_anchor = &C[i * ldc];
                            for (int j = 0; j < N; ++j) {
                                *C_anchor += alpha * inline_dot<T_IN, T_OUT>(K, &A[i], lda, &B[j], ldb);
                                C_anchor++;
                            }
                TS_PARALLEL_FOR_END()
                    break;
                case 2: // A: NoTrans, B: Trans
                TS_PARALLEL_FOR_BEGIN(i, 0, M)
                            T_OUT *C_anchor = &C[i * ldc];
                            for (int j = 0; j < N; ++j) {
                                *C_anchor += alpha * inline_dot<T_IN, T_OUT>(K, &A[i * lda], 1, &B[j * ldb], 1);
                                C_anchor++;
                            }
                TS_PARALLEL_FOR_END()
                    break;
                default: // A: Trans, B: Trans
                TS_PARALLEL_FOR_BEGIN(i, 0, M)
                            T_OUT *C_anchor = &C[i * ldc];
                            for (int j = 0; j < N; ++j) {
                                *C_anchor += alpha * inline_dot<T_IN, T_OUT>(K, &A[i], lda, &B[j * ldb], 1);
                                C_anchor++;
                            }
                TS_PARALLEL_FOR_END()
                    break;
            }
        }

        // TODO: it has deviation in some case, when N, M, K is large
        template<typename T_IN,typename T_OUT>
        void
        math<T_IN,T_OUT>::gemm(
                blas::Order Order,
                blas::Transpose TransA,
                blas::Transpose TransB,
                int M, int N, int K,
                T_IN alpha,
                const T_IN *A, int lda,
                const T_IN *B, int ldb,
                T_IN beta,
                T_OUT *C, int ldc) {
            if (Order == blas::ColMajor) {
                inline_gemm_row_major<T_IN, T_OUT>(TransB, TransA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
            } else {
                inline_gemm_row_major<T_IN, T_OUT>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }

        template<typename T_IN,typename T_OUT>
        T_OUT math<T_IN,T_OUT>::dot(int N, const T_IN *x, const T_IN *y) {
            return dot(N, x, 1, y, 1);
        }

        template<typename T_IN, typename T_OUT>
        void math<T_IN, T_OUT>::gemm(blas::Transpose TransA, blas::Transpose TransB, int M, int N, int K, T_IN alpha, const T_IN *A,
                           const T_IN *B, T_IN beta, T_OUT *C) {
            int lda = (TransA == blas::NoTrans ? K : M);
            int ldb = (TransB == blas::NoTrans ? N : K);
            int ldc = N;
            inline_gemm_row_major<T_IN, T_OUT>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template<typename T_IN, typename T_OUT>
        void math<T_IN, T_OUT>::pack8_A(int row, int col, const T_IN *from, int lda, T_IN *to) {
            int out_loop = row >> 3;
            int remain = out_loop << 3;

            //T_OUT* to_at = to;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++) {
                int n = nn * 8;
                const T_IN* k0 = from + n * lda;
                const T_IN* k1 = k0 + lda;
                const T_IN* k2 = k1 + lda;
                const T_IN* k3 = k2 + lda;
                const T_IN* k4 = k3 + lda;
                const T_IN* k5 = k4 + lda;
                const T_IN* k6 = k5 + lda;
                const T_IN* k7 = k6 + lda;

                T_IN* to_at = to + n * col;

                for (int i = 0; i < col; i++) {
                    *to_at++ = *k0++;
                    *to_at++ = *k1++;
                    *to_at++ = *k2++;
                    *to_at++ = *k3++;
                    *to_at++ = *k4++;
                    *to_at++ = *k5++;
                    *to_at++ = *k6++;
                    *to_at++ = *k7++;
                }
            }

            //NOTE:Maybe i should pack 4x4 on remain size
            //to_at = to + remain * col;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < row; n++) {
                const T_IN* k0 = from + n * lda;
                T_IN* to_at = to + n * col;
                for (int i = 0; i < col; i++) {
                    *to_at++ = *k0++;
                }
            }
        }

        template<typename T_IN, typename T_OUT>
        inline void inline_pack8_B(int row, int col, const T_IN *from, int ldb, T_IN *to) {
            int out_loop = col >> 3;
            int remain = out_loop << 3;

            //T_OUT* to_at = to;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++) {
                int n = nn * 8;
                const T_IN* from_at = from + n;
                T_IN* to_at = to + n * row;

                for (int i = 0; i < row; i++) {
                    *to_at++ = from_at[0];
                    *to_at++ = from_at[1];
                    *to_at++ = from_at[2];
                    *to_at++ = from_at[3];
                    *to_at++ = from_at[4];
                    *to_at++ = from_at[5];
                    *to_at++ = from_at[6];
                    *to_at++ = from_at[7];

                    from_at += ldb;
                }
            }

            //to_at = to + remain * row;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < col; n++) {
                const T_IN* from_at = from + n;
                T_IN* to_at = to + n * row;

                for (int i = 0; i < row; i++) {
                    *to_at++ = from_at[0];
                    from_at += ldb;
                }
            }
        }

        template<>
        inline void inline_pack8_B<float, float>(int row, int col, const float *from, int ldb, float *to) {
            int out_loop = col >> 3;
            int remain = out_loop << 3;

            //float* to_at = to;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int nn = 0; nn < out_loop; nn++) {
                int n = nn * 8;
                const float* from_at = from + n;
                float* to_at = to + n * row;

                for (int i = 0; i < row; i++) {
                    float32x4x2 from_at_x4x2(from_at);
                    from_at_x4x2.store(to_at);

                    from_at += ldb;
                    to_at += 8;
                }
            }

            //to_at = to + remain * row;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int n = remain; n < col; n++) {
                const float* from_at = from + n;
                float* to_at = to + n * row;

                for (int i = 0; i < row; i++) {
                    *to_at++ = from_at[0];
                    from_at += ldb;
                }
            }
        }

        template<typename T_IN, typename T_OUT>
        void math<T_IN, T_OUT>::pack8_B(int row, int col, const T_IN *from, int ldb, T_IN *to) {
            inline_pack8_B<T_IN, T_OUT>(row, col, from, ldb, to);
        }

        template<typename T_IN, typename T_OUT>
        inline void kernel_8x8(int M, int K, int N, T_IN alpha, const T_IN *A, const T_IN *B, T_IN beta, T_OUT *C, int ldc) {

        }

        template<>
        inline void kernel_8x8<float, float>(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C, int ldc) {
            const float* p_A = A;
            const float* p_B = B;
            float* p_C = C;

            int out_loop = M >> 3;
            int remain = out_loop << 3;
            float* output_at = p_C;
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int mm = 0; mm < out_loop; mm++) {
                int m = mm * 8;
                float* output_row0 = output_at + m * ldc;
                float* output_row1 = output_row0 + ldc;
                float* output_row2 = output_row1 + ldc;
                float* output_row3 = output_row2 + ldc;
                float* output_row4 = output_row3 + ldc;
                float* output_row5 = output_row4 + ldc;
                float* output_row6 = output_row5 + ldc;
                float* output_row7 = output_row6 + ldc;

                const float* A_store = p_A + m * K;

                int n_loop = N >> 3;
                int n_remain = n_loop << 3;
                for (int nn = 0; nn < n_loop; nn++)
                {
                    int n = nn * 8;

                    const float* A_at = A_store;
                    const float* B_at = p_B + n * K;

                    float32x4x2 c0(0.f), c1(0.f), c2(0.f), c3(0.f);
                    float32x4x2 c4(0.f), c5(0.f), c6(0.f), c7(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++) {
                        //=====================pack_gemm k==0=====================
                        float32x4x2 k0 = broadcast2float32x4x2(A_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 k1 = broadcast2float32x4x2(A_at + 1);   //[k10,k10,k10,k10,k10,k10,k10,k10]
                        float32x4x2 k2 = broadcast2float32x4x2(A_at + 2);   //[k20,k20,k20,k20,k20,k20,k20,k20]
                        float32x4x2 k3 = broadcast2float32x4x2(A_at + 3);   //[k30,k30,k30,k30,k30,k30,k30,k30]

                        float32x4x2 a0(B_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]

                        c0 = fmadd(a0, k0, c0);
                        c1 = fmadd(a0, k1, c1);
                        c2 = fmadd(a0, k2, c2);
                        c3 = fmadd(a0, k3, c3);
                        //Note:The number of registers is limited
                        k0 = broadcast2float32x4x2(A_at + 4);               //[k40,k40,k40,k40,k40,k40,k40,k40]
                        k1 = broadcast2float32x4x2(A_at + 5);               //[k50,k50,k50,k50,k50,k50,k50,k50]
                        k2 = broadcast2float32x4x2(A_at + 6);               //[k60,k60,k60,k60,k60,k60,k60,k60]
                        k3 = broadcast2float32x4x2(A_at + 7);               //[k70,k70,k70,k70,k70,k70,k70,k70]

                        c4 = fmadd(a0, k0, c4);
                        c5 = fmadd(a0, k1, c5);
                        c6 = fmadd(a0, k2, c6);
                        c7 = fmadd(a0, k3, c7);

                        //=====================pack_gemm k==1=====================
                        k0 = broadcast2float32x4x2(A_at + 8);               //[k01,k01,k01,k01,k01,k01,k01,k01]
                        k1 = broadcast2float32x4x2(A_at + 9);               //[k11,k11,k11,k11,k11,k11,k11,k11]
                        k2 = broadcast2float32x4x2(A_at + 10);              //[k21,k21,k21,k21,k21,k21,k21,k21]
                        k3 = broadcast2float32x4x2(A_at + 11);              //[k31,k31,k31,k31,k31,k31,k31,k31]

                        float32x4x2 a1(B_at + 8);                              //[a10,a11,a12,a13,a14,a15,a16,a17]

                        c0 = fmadd(a1, k0, c0);
                        c1 = fmadd(a1, k1, c1);
                        c2 = fmadd(a1, k2, c2);
                        c3 = fmadd(a1, k3, c3);

                        k0 = broadcast2float32x4x2(A_at + 12);              //[k41,k41,k41,k41,k41,k41,k41,k41]
                        k1 = broadcast2float32x4x2(A_at + 13);              //[k51,k51,k51,k51,k51,k51,k51,k51]
                        k2 = broadcast2float32x4x2(A_at + 14);              //[k61,k61,k61,k61,k61,k61,k61,k61]
                        k3 = broadcast2float32x4x2(A_at + 15);              //[k71,k71,k71,k71,k71,k71,k71,k71]

                        c4 = fmadd(a1, k0, c4);
                        c5 = fmadd(a1, k1, c5);
                        c6 = fmadd(a1, k2, c6);
                        c7 = fmadd(a1, k3, c7);
                        //=====================pack_gemm k==2=====================
                        k0 = broadcast2float32x4x2(A_at + 16);              //[k02,k02,k02,k02,k02,k02,k02,k02]
                        k1 = broadcast2float32x4x2(A_at + 17);              //[k12,k12,k12,k12,k12,k12,k12,k12]
                        k2 = broadcast2float32x4x2(A_at + 18);              //[k22,k21,k21,k21,k21,k21,k21,k21]
                        k3 = broadcast2float32x4x2(A_at + 19);              //[k32,k32,k32,k32,k32,k32,k32,k32]

                        float32x4x2 a2(B_at + 16);                             //[a20,a21,a22,a23,a24,a25,a26,a27]

                        c0 = fmadd(a2, k0, c0);
                        c1 = fmadd(a2, k1, c1);
                        c2 = fmadd(a2, k2, c2);
                        c3 = fmadd(a2, k3, c3);

                        k0 = broadcast2float32x4x2(A_at + 20);              //[k42,k42,k42,k42,k42,k42,k42,k42]
                        k1 = broadcast2float32x4x2(A_at + 21);              //[k52,k52,k52,k52,k52,k52,k52,k52]
                        k2 = broadcast2float32x4x2(A_at + 22);              //[k62,k62,k62,k62,k62,k62,k62,k62]
                        k3 = broadcast2float32x4x2(A_at + 23);              //[k72,k72,k72,k72,k72,k72,k72,k72]

                        c4 = fmadd(a2, k0, c4);
                        c5 = fmadd(a2, k1, c5);
                        c6 = fmadd(a2, k2, c6);
                        c7 = fmadd(a2, k3, c7);
                        //=====================pack_gemm k==3=====================
                        k0 = broadcast2float32x4x2(A_at + 24);              //[k03,k03,k03,k03,k03,k03,k03,k03]
                        k1 = broadcast2float32x4x2(A_at + 25);              //[k13,k13,k13,k13,k13,k13,k13,k13]
                        k2 = broadcast2float32x4x2(A_at + 26);              //[k23,k23,k23,k23,k23,k23,k23,k23]
                        k3 = broadcast2float32x4x2(A_at + 27);              //[k33,k33,k33,k33,k33,k33,k33,k33]

                        float32x4x2 a3(B_at + 24);                             //[a30,a31,a32,a33,a34,a35,a36,a37]

                        c0 = fmadd(a3, k0, c0);
                        c1 = fmadd(a3, k1, c1);
                        c2 = fmadd(a3, k2, c2);
                        c3 = fmadd(a3, k3, c3);

                        k0 = broadcast2float32x4x2(A_at + 28);              //[k43,k43,k43,k43,k43,k43,k43,k43]
                        k1 = broadcast2float32x4x2(A_at + 29);              //[k53,k53,k53,k53,k53,k53,k53,k53]
                        k2 = broadcast2float32x4x2(A_at + 30);              //[k63,k63,k63,k63,k63,k63,k63,k63]
                        k3 = broadcast2float32x4x2(A_at + 31);              //[k73,k73,k73,k73,k73,k73,k73,k73]

                        c4 = fmadd(a3, k0, c4);
                        c5 = fmadd(a3, k1, c5);
                        c6 = fmadd(a3, k2, c6);
                        c7 = fmadd(a3, k3, c7);

                        A_at += 32;
                        B_at += 32;
                    }

                    for (int k = k_remain; k < K; k++) {
                        float32x4x2 k0 = broadcast2float32x4x2(A_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 k1 = broadcast2float32x4x2(A_at + 1);   //[k10,k10,k10,k10,k10,k10,k10,k10]
                        float32x4x2 k2 = broadcast2float32x4x2(A_at + 2);   //[k20,k20,k20,k20,k20,k20,k20,k20]
                        float32x4x2 k3 = broadcast2float32x4x2(A_at + 3);   //[k30,k30,k30,k30,k30,k30,k30,k30]

                        float32x4x2 a0(B_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]

                        c0 = fmadd(a0, k0, c0);
                        c1 = fmadd(a0, k1, c1);
                        c2 = fmadd(a0, k2, c2);
                        c3 = fmadd(a0, k3, c3);

                        k0 = broadcast2float32x4x2(A_at + 4);               //[k40,k40,k40,k40,k40,k40,k40,k40]
                        k1 = broadcast2float32x4x2(A_at + 5);               //[k50,k50,k50,k50,k50,k50,k50,k50]
                        k2 = broadcast2float32x4x2(A_at + 6);               //[k60,k60,k60,k60,k60,k60,k60,k60]
                        k3 = broadcast2float32x4x2(A_at + 7);               //[k70,k70,k70,k70,k70,k70,k70,k70]

                        c4 = fmadd(a0, k0, c4);
                        c5 = fmadd(a0, k1, c5);
                        c6 = fmadd(a0, k2, c6);
                        c7 = fmadd(a0, k3, c7);

                        A_at += 8;
                        B_at += 8;
                    }

                    c0.store(output_row0); c1.store(output_row1);
                    c2.store(output_row2); c3.store(output_row3);
                    c4.store(output_row4); c5.store(output_row5);
                    c6.store(output_row6); c7.store(output_row7);

                    output_row0 += 8; output_row1 += 8;
                    output_row2 += 8; output_row3 += 8;
                    output_row4 += 8; output_row5 += 8;
                    output_row6 += 8; output_row7 += 8;
                }

                for (int n = n_remain; n < N; n++)
                {
                    const float* A_at = A_store;
                    const float* B_at = p_B + n * K;
                    float32x4x2 sum_col0(0.f), sum_col1(0.f), sum_col2(0.f), sum_col3(0.f);
                    float32x4x2 sum_col(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++) {
                        // int k = kk * 4;

                        float32x4x2 a0 = broadcast2float32x4x2(B_at);          //[a00,a00,a00,a00,a00,a00,a00,a00]
                        float32x4x2 a1 = broadcast2float32x4x2(B_at + 1);      //[a10,a10,a10,a10,a10,a10,a10,a10]
                        float32x4x2 a2 = broadcast2float32x4x2(B_at + 2);      //[a20,a20,a20,a20,a20,a20,a20,a20]
                        float32x4x2 a3 = broadcast2float32x4x2(B_at + 3);      //[a30,a30,a30,a30,a30,a30,a30,a30]

                        float32x4x2 k0(A_at);                               //[k00,k10,k20,k30,k40,k50,k60,k70]
                        float32x4x2 k1(A_at + 8);                           //[k01,k11,k21,k31,k41,k51,k61,k71]
                        float32x4x2 k2(A_at + 16);                          //[k02,k12,k22,k32,k42,k52,k62,k72]
                        float32x4x2 k3(A_at + 24);                          //[k03,k13,k23,k33,k43,k53,k63,k73]

                        sum_col0 = fmadd(k0, a0, sum_col0);
                        sum_col1 = fmadd(k1, a1, sum_col1);
                        sum_col2 = fmadd(k2, a2, sum_col2);
                        sum_col3 = fmadd(k3, a3, sum_col3);

                        A_at += 32;
                        B_at += 4;
                    }

                    sum_col0 += sum_col1;
                    sum_col2 += sum_col3;
                    sum_col += sum_col0;
                    sum_col += sum_col2;

                    for (int k = k_remain; k < K; k++) {
                        float32x4x2 a0 = broadcast2float32x4x2(B_at);          //[a00,a00,a00,a00,a00,a00,a00,a00]
                        float32x4x2 k0(A_at);                               //[k00,k10,k20,k30,k40,k50,k60,k70]

                        sum_col = fmadd(k0, a0, sum_col);

                        A_at += 8;
                        B_at += 1;
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
            for (int m = remain; m < M; m++) {
                float* output_row0 = output_at + m * ldc;
                const float* A_store = p_A + m * K;

                int n_loop = N >> 3;
                int n_remain = n_loop << 3;
                for (int nn = 0; nn < n_loop; nn++) {
                    int n = nn * 8;

                    const float* A_at = A_store;
                    const float* B_at = p_B + n * K;

                    float32x4x2 c0(0.f);

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++) {

                        float32x4x2 k0 = broadcast2float32x4x2(A_at);       //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 k1 = broadcast2float32x4x2(A_at + 1);   //[k01,k01,k01,k01,k01,k01,k01,k01]
                        float32x4x2 k2 = broadcast2float32x4x2(A_at + 2);   //[k02,k02,k02,k02,k02,k02,k02,k02]
                        float32x4x2 k3 = broadcast2float32x4x2(A_at + 3);   //[k03,k03,k03,k03,k03,k03,k03,k03]

                        float32x4x2 a0(B_at);                                  //[a00,a01,a02,a03,a04,a05,a06,a07]
                        float32x4x2 a1(B_at + 8);                              //[a10,a11,a12,a13,a14,a15,a16,a17]
                        float32x4x2 a2(B_at + 16);                             //[a20,a21,a22,a23,a24,a25,a26,a27]
                        float32x4x2 a3(B_at + 24);                             //[a30,a31,a32,a33,a34,a35,a36,a37]

                        c0 = fmadd(k0, a0, c0);
                        c0 = fmadd(k1, a1, c0);
                        c0 = fmadd(k2, a2, c0);
                        c0 = fmadd(k3, a3, c0);

                        A_at += 4;
                        B_at += 32;
                    }

                    for (int k = k_remain; k < K; k++) {
                        float32x4x2 k0 = broadcast2float32x4x2(A_at);        //[k00,k00,k00,k00,k00,k00,k00,k00]
                        float32x4x2 a0(B_at);                                   //[a00,a01,a02,a03,a04,a05,a06,a07]

                        c0 = fmadd(k0, a0, c0);

                        A_at += 1;
                        B_at += 8;
                    }

                    c0.store(output_row0);
                    output_row0 += 8;
                }

                for (int n = n_remain; n < N; n++) {
                    float32x4 c0(0.f);
                    float sum0 = 0;

                    const float* A_at = A_store;
                    const float* B_at = p_B + n * K;

                    int k_loop = K >> 2;
                    int k_remain = k_loop << 2;
                    for (int kk = 0; kk < k_loop; kk++) {
                        // int k = kk * 4;
                        float32x4 k0(A_at);
                        float32x4 a0(B_at);

                        c0 = fmadd(k0, a0, c0);

                        A_at += 4;
                        B_at += 4;
                    }

                    sum0 = ts::sum(c0);

                    for (int k = k_remain; k < K; k++) {
                        sum0 += (*A_at) * (*B_at);
                        A_at++;
                        B_at++;
                    }

                    *output_row0 = sum0;
                    output_row0++;
                }
            }
        }

        template<typename T_IN, typename T_OUT>
        void math<T_IN, T_OUT>::gemm(int M, int N, int K, T_IN alpha, const T_IN *A, const T_IN *B,
                                     T_IN beta, T_OUT *C, bool A_need_pack, bool B_need_pack) {
            Tensor A_packed;
            Tensor B_packed;
            if (A_need_pack) {
                A_packed = Tensor(Tensor::InFlow::HOST, dtypeid<T_IN>::id, {int32_t(M * K),});
            }
            if (B_need_pack) {
                B_packed = Tensor(Tensor::InFlow::HOST, dtypeid<T_IN>::id, {int32_t(N * K),});
            }
            self::gemm(M, N, K, alpha, A, A_packed.data<T_IN>(), B, B_packed.data<T_IN>(), beta, C, A_need_pack, B_need_pack);
        }

        template<typename T_IN, typename T_OUT>
        void math<T_IN, T_OUT>::gemm(int M, int N, int K, T_IN alpha, const T_IN *A, T_IN *A_packed, const T_IN *B, T_IN *B_packed,
                           T_IN beta, T_OUT *C, bool A_need_pack, bool B_need_pack) {

            if (!ts::near(alpha, T_IN(1)) || !ts::near(beta, T_IN(0))) {
                TS_LOG_ERROR << "alpha should be one and beta should be zero now!"<< eject;
            }

            if (A_need_pack) {
                math<T_IN, T_OUT>::pack8_A(M, K, A, K, A_packed);
            }
            if (B_need_pack) {
                math<T_IN, T_OUT>::pack8_B(K, N, B, N, B_packed);
            }

            if (A_need_pack && B_need_pack) {
                kernel_8x8<T_IN, T_OUT>(M, K, N, alpha, A_packed, B_packed, beta, C, N);
            }
            else if (A_need_pack && !B_need_pack) {
                kernel_8x8<T_IN, T_OUT>(M, K, N, alpha, A_packed, B, beta, C, N);
            }
            else if (!A_need_pack && B_need_pack) {
                kernel_8x8<T_IN, T_OUT>(M, K, N, alpha, A, B_packed, beta, C, N);
            }
            else {
                kernel_8x8<T_IN, T_OUT>(M, K, N, alpha, A, B, beta, C, N);
            }       
        }


        template<typename T_IN, typename T_OUT>
        inline T_OUT inline_asum(int N, const T_IN *x, int incx) {
            T_OUT sum = 0;
            // block: 4
            int i = 0;
            static const int block_size = 4;
            int blocked_N = N % block_size ? N - block_size : N;
            for (; i < blocked_N; i += block_size) {
                sum += abs(*x); x += incx;
                sum += abs(*x); x += incx;
                sum += abs(*x); x += incx;
                sum += abs(*x); x += incx;
            }
            for (; i < N; ++i) {
                sum += abs(*x); x += incx;
            }
            return sum;
        }

        template<typename T_IN, typename T_OUT>
        T_OUT math<T_IN, T_OUT>::asum(int N, const T_IN *x, int incx) {
            std::vector<T_OUT> parallel_sum(TS_PARALLEL_SIZE, T_OUT(0));
            TS_PARALLEL_RANGE_BEGIN(range, 0, N)
                    const T_IN *xx = x + range.first * incx;
                    const auto count = range.second - range.first;
                    parallel_sum[__parallel_id] += inline_asum<T_IN, T_OUT>(count, xx, incx);
            TS_PARALLEL_RANGE_END()
            T_OUT sum = 0;
            for (auto value : parallel_sum) sum += value;
            return sum;
        }

        template<typename T_IN, typename T_OUT>
        T_OUT math<T_IN, T_OUT>::abs(T_IN val) {
            return T_OUT(std::fabs(val));
        }

        template<typename T_IN, typename T_OUT>
        void math<T_IN, T_OUT>::matrix_transpose(const T_IN* A, T_OUT* B, int m, int n) {
            int i, j;
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    B[i*m + j] = A[j*n + i];
                }
            }
        }
    }
}

template class ts::cpu::math<ts::dtype<ts::FLOAT32>::declare, ts::dtype<ts::FLOAT32>::declare>;
template class ts::cpu::math<ts::dtype<ts::FLOAT64>::declare, ts::dtype<ts::FLOAT64>::declare>;
template class ts::cpu::math<ts::dtype<ts::INT8>::declare, ts::dtype<ts::INT32>::declare>;