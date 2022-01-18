#ifndef TENSORSTACK_KERNELS_GPU_MATH_CUBLAS_H
#define TENSORSTACK_KERNELS_GPU_MATH_CUBLAS_H

#include "core/tensor.h"
#include "../common/blas.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace ts {
    namespace gpu {
        namespace cublas {
            template <typename T>
            class TS_DEBUG_API math {
            public:
                static void check(const Tensor &tensor) {
                    if (tensor.device().type() != CUBLAS) throw DeviceMismatchException(Device(CUBLAS), tensor.device());
                }

                static bool dot(cublasHandle_t hadnle, const int N, const T *x, const T *y, T * z);

                static bool gemm(
                    cublasHandle_t hadnle,
                    Order Order,
                    Transpose TransA,
                    Transpose TransB,
                    int M, int N, int K,
                    T alpha,
                    const T *A, int lda,
                    const T *B, int ldb,
                    T beta,
                    T *C, int ldc);

                static bool gemm(
                    cublasHandle_t hadnle,
                    cublas::Transpose TransA,
                    cublas::Transpose TransB,
                    int M, int N, int K,
                    T alpha,
                    const T *A,
                    const T *B,
                    T beta,
                    T *C);

                static bool asum(
                    cublasHandle_t hadnle,
                    int N,
                    const T *x,
                    T *out
                );
            };
        }
    }
}

extern template class ts::gpu::cublas::math<half>;
extern template class ts::gpu::cublas::math<ts::dtype<ts::FLOAT32>::declare>;
extern template class ts::gpu::cublas::math<ts::dtype<ts::FLOAT64>::declare>;


#endif //TENSORSTACK_KERNELS_GPU_MATH_CUBLAS_H
