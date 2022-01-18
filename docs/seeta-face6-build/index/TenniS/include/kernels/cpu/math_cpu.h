//
// Created by kier on 2018/7/19.
//

#ifndef TENSORSTACK_KERNELS_CPU_MATH_CPU_H
#define TENSORSTACK_KERNELS_CPU_MATH_CPU_H

#include "core/tensor.h"
#include "../common/blas.h"

namespace ts {
    namespace cpu {
        template <typename T_IN, typename T_OUT>
        class TS_DEBUG_API math {
        public:
            using self = math;

            static void check(const Tensor &tensor) {
                if (tensor.device().type() != CPU) throw DeviceMismatchException(Device(CPU), tensor.device());
            }

            static T_OUT abs(T_IN val);

            static T_OUT dot(
                    int N,
                    const T_IN *x,
                    int incx,
                    const T_IN *y,
                    int incy
                    );

            static T_OUT dot(int N, const T_IN *x, const T_IN *y);

            static void gemm(
                    blas::Order Order,
                    blas::Transpose TransA,
                    blas::Transpose TransB,
                    int M, int N, int K,
                    T_IN alpha,
                    const T_IN *A, int lda,
                    const T_IN *B, int ldb,
                    T_IN beta,
                    T_OUT *C, int ldc);

            static void gemm(
                    blas::Transpose TransA,
                    blas::Transpose TransB,
                    int M, int N, int K,
                    T_IN alpha, const T_IN *A, const T_IN *B,
                    T_IN beta, T_OUT *C);

            static void pack8_A(int row, int col, const T_IN *from, int lda, T_IN *to);

            static void pack8_B(int row, int col, const T_IN *from, int ldb, T_IN *to);

            static void gemm(
                    int M, int N, int K,
                    T_IN alpha, const T_IN *A,
                    const T_IN *B,
                    T_IN beta, T_OUT *C,
                    bool A_need_pack, bool B_need_pack);

            //NOTE:for pack gemm.
            //only support row major now and no trans
            //alpha==1,beta==0 should be promised now
            //input and output should be float now
            static void gemm(
                    int M, int N, int K,
                    T_IN alpha, const T_IN *A, T_IN *A_packed,
                    const T_IN *B, T_IN *B_packed,
                    T_IN beta, T_OUT *C,
                    bool A_need_pack, bool B_need_pack);

            static T_OUT asum(
                    int N,
                    const T_IN *x,
                    int incx
                    );

            static void matrix_transpose(const T_IN* A, T_OUT* B, int m, int n);
        };
    }
}

extern template class ts::cpu::math<ts::dtype<ts::FLOAT32>::declare, ts::dtype<ts::FLOAT32>::declare>;
extern template class ts::cpu::math<ts::dtype<ts::FLOAT64>::declare, ts::dtype<ts::FLOAT64>::declare>;
extern template class ts::cpu::math<ts::dtype<ts::INT8>::declare, ts::dtype<ts::INT32>::declare>;


#endif //TENSORSTACK_KERNELS_CPU_MATH_CPU_H
