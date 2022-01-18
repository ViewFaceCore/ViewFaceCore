//
// Created by kier on 2019/1/26.
//

#include "kernels/cblas/math_cblas.h"
#include "utils/platform.h"

#if TS_PLATFORM_OS_MAC || TS_PLATFORM_OS_IOS
#include <Accelerate/Accelerate.h>
#elif TS_PLATFORM_OS_LINUX
#include <openblas/cblas.h>
#elif TS_PLATFORM_OS_WINDOWS && TS_PLATFORM_CC_MINGW
#include <OpenBLAS/cblas.h>
#else
#include <cblas.h>
#endif

#include <kernels/cpu/math_cpu.h>

namespace ts {
    namespace cblas {
        template<typename T>
        T math<T>::abs(T val) {
            return std::fabs(val);
        }

        template<typename T>
        inline T cblas_dot(int N, const T *x, int incx, const T *y, int incy) {
            return cpu::math<T, T>::dot(N, x, incx, y, incy);
        }

        template <>
        inline float cblas_dot<float>(int N, const float *x, int incx, const float *y, int incy) {
            return cblas_sdot(N, x, incx, y, incy);
        }

        template <>
        inline double cblas_dot<double>(int N, const double *x, int incx, const double *y, int incy) {
            return cblas_ddot(N, x, incx, y, incy);
        }

        template<typename T>
        T math<T>::dot(int N, const T *x, int incx, const T *y, int incy) {
            return cblas_dot<T>(N, x, incx, y, incy);
        }

        template<typename T>
        T math<T>::dot(int N, const T *x, const T *y) {
            return dot(N, x, 1, y, 1);
        }

        template <typename T>
        void cblas_gemm(blas::Order Order, blas::Transpose TransA, blas::Transpose TransB,
                        int M, int N, int K,
                        T alpha, const T *A, int lda,
                        const T *B, int ldb, T beta,
                        T *C, int ldc) {
            cpu::math<T, T>::gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template <>
        void cblas_gemm<float>(blas::Order Order, blas::Transpose TransA, blas::Transpose TransB,
                               int M, int N, int K,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta,
                               float *C, int ldc) {
            CBLAS_ORDER cblas_order = Order == ts::blas::RowMajor ? CblasRowMajor : CblasColMajor;
            CBLAS_TRANSPOSE cblas_TransA = TransA == ts::blas::NoTrans ? CblasNoTrans : CblasTrans;
            CBLAS_TRANSPOSE cblas_TransB = TransB == ts::blas::NoTrans ? CblasNoTrans : CblasTrans;
            cblas_sgemm(cblas_order, cblas_TransA, cblas_TransB,
                    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template <>
        void cblas_gemm<double>(blas::Order Order, blas::Transpose TransA, blas::Transpose TransB,
                                int M, int N, int K,
                                double alpha, const double *A, int lda,
                                const double *B, int ldb, double beta,
                                double *C, int ldc) {
            CBLAS_ORDER cblas_order = Order == ts::blas::RowMajor ? CblasRowMajor : CblasColMajor;
            CBLAS_TRANSPOSE cblas_TransA = TransA == ts::blas::NoTrans ? CblasNoTrans : CblasTrans;
            CBLAS_TRANSPOSE cblas_TransB = TransB == ts::blas::NoTrans ? CblasNoTrans : CblasTrans;
            cblas_dgemm(cblas_order, cblas_TransA, cblas_TransB,
                        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template<typename T>
        void
        math<T>::gemm(blas::Order Order, blas::Transpose TransA, blas::Transpose TransB,
                      int M, int N, int K,
                      T alpha, const T *A, int lda,
                      const T *B, int ldb, T beta,
                      T *C, int ldc) {
            cblas_gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template<typename T>
        void math<T>::gemm(blas::Transpose TransA, blas::Transpose TransB, int M, int N, int K, T alpha, const T *A,
                           const T *B, T beta, T *C) {
            blas::Order Order = blas::RowMajor;
            int lda = (TransA == blas::NoTrans ? K : M);
            int ldb = (TransB == blas::NoTrans ? N : K);
            int ldc = N;
            gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template <typename T>
        T cblas_asum(int N, const T *x, int incx) {
            return cpu::math<T, T>::asum(N, x, incx);
        }

        template <>
        float cblas_asum<float>(int N, const float *x, int incx) {
            return cblas_sasum(N, x, incx);
        }

        template <>
        double cblas_asum<double>(int N, const double *x, int incx) {
            return cblas_dasum(N, x, incx);
        }

        template<typename T>
        T math<T>::asum(int N, const T *x, int incx) {
            return cblas_asum(N, x, incx);
        }
    }
}

template class ts::cblas::math<ts::dtype<ts::FLOAT32>::declare>;
template class ts::cblas::math<ts::dtype<ts::FLOAT64>::declare>;
