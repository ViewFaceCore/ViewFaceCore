#ifndef TENSORSTACK_KERNELS_GPU_MATH_GPU_H
#define TENSORSTACK_KERNELS_GPU_MATH_GPU_H

#include "core/tensor.h"
#include "../common/blas.h"

namespace ts {
    namespace gpu {
        template <typename T>
        class TS_DEBUG_API math {
        public:
            static void check(const Tensor &tensor) {
                if (tensor.device().type() != GPU) throw DeviceMismatchException(Device(GPU), tensor.device());
            }

            //static bool dot(const int N, const T *x, int incx, const T *y, int incy, T * z);

            static bool dot(int N, const T *x, const T *y, T *z);

            static bool gemm(
                cublas::Order Order,
                cublas::Transpose TransA,
                cublas::Transpose TransB,
                int M, int N, int K,
                T alpha,
                const T *A, int lda,
                const T *B, int ldb,
                T beta,
                T *C, int ldc);

            static bool gemm(
                cublas::Transpose TransA,
                cublas::Transpose TransB,
                int M, int N, int K,
                T alpha,
                const T *A,
                const T *B,
                T beta,
                T *C);

            static bool asum(
                int N,
                const T *x,
                T* out
            );

            static bool sum(
                int N,
                const T *x,
                T* out
            );
        };
    }
}

extern template class ts::gpu::math<ts::dtype<ts::FLOAT32>::declare>;
extern template class ts::gpu::math<ts::dtype<ts::FLOAT64>::declare>;


#endif //TENSORSTACK_KERNELS_GPU_MATH_GPU_H
