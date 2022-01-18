//
// Created by kier on 2019/1/26.
//

#ifndef TENSORSTACK_KERNELS_CBLAS_MATH_CBLAS_H
#define TENSORSTACK_KERNELS_CBLAS_MATH_CBLAS_H

#include "core/tensor.h"
#include "../common/blas.h"

namespace ts {
    namespace cblas {
        template <typename T>
        class TS_DEBUG_API math {
        public:
            static void check(const Tensor &tensor) {
                if (tensor.device().type() != CPU) throw DeviceMismatchException(Device(CPU), tensor.device());
            }

            static T abs(T val);

            static T dot(
                    int N,
                    const T *x,
                    int incx,
                    const T *y,
                    int incy
                    );

            static T dot(int N, const T *x, const T *y);

            static void gemm(
                    blas::Order Order,
                    blas::Transpose TransA,
                    blas::Transpose TransB,
                    int M, int N, int K,
                    T alpha,
                    const T *A, int lda,
                    const T *B, int ldb,
                    T beta,
                    T *C, int ldc);

            static void gemm(
                    blas::Transpose TransA,
                    blas::Transpose TransB,
                    int M, int N, int K,
                    T alpha, const T *A, const T *B,
                    T beta, T *C);

            static T asum(
                    int N,
                    const T *x,
                    int incx
                    );
        };
    }
}

extern template class ts::cblas::math<ts::dtype<ts::FLOAT32>::declare>;
extern template class ts::cblas::math<ts::dtype<ts::FLOAT64>::declare>;


#endif //TENSORSTACK_KERNELS_CBLAS_MATH_CBLAS_H
