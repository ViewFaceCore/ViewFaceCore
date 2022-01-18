//
// Created by kier on 2019/2/18.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_INNER_PROD_H
#define TENSORSTACK_BACKEND_BASE_BASE_INNER_PROD_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class InnerProd : public OperatorOnDevice {
        public:
            using self = InnerProd;
            using supper = OperatorOnDevice;

            InnerProd();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param lhs
             * @param rhs
             * @param out
             * @note all tensor's dtype is same, and all tensors' memory device are give in constructor
             */
            virtual void inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out, Stack &stack, bool kernel_packed) {
                if (kernel_packed) {
                    TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
                }
                inner_prod(lhs, rhs, transpose, out);
            }

            /**
             *
             * @param lhs
             * @param rhs
             * @param out
             * @note all tensor's dtype is same, and all tensors' memory device are give in constructor
             */
            virtual void inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out) {
                TS_LOG_ERROR << "What a Terrible Failure: not implement inner_prod core." << eject;
            }

        private:
            bool m_transpose = false;
            bool m_kernel_packed = false;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_INNER_PROD_H
