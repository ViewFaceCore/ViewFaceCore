//
// Created by kier on 2019/3/5.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_GEMM_H
#define TENSORSTACK_BACKEND_BASE_BASE_GEMM_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Gemm : public OperatorOnDevice {
        public:
            using self = Gemm;
            using supper = OperatorOnDevice;

            Gemm();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param A The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
             * @param B The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
             * @param C The shape of C should be unidirectional broadcastable to (M, N), means (1 or M, 1 or N).
             * @param K
             * @param alpha
             * @param beta
             * @param transA
             * @param transB
             * @param out Output tensor of shape (M, N).
             * do Row major GEMM,
             */
            virtual void gemm(const Tensor &A, const Tensor &B, const Tensor &C, int K,
                              float alpha, float beta, bool transA, bool transB, Tensor &out) = 0;

        private:
            float m_alpha = 1.0f;
            float m_beta = 1.0f;
            bool m_transA = false;
            bool m_transB = false;
        };
    }

}

#endif //TENSORSTACK_BACKEND_BASE_BASE_GEMM_H
