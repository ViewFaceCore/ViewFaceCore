//
// Created by kier on 2019/4/8.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_STACK_H
#define TENSORSTACK_BACKEND_BASE_BASE_STACK_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class StackTensor : public OperatorOnDevice {
        public:
            using self = StackTensor;
            using supper = OperatorOnDevice;

            StackTensor();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void stack_tensor(const std::vector<Tensor> &x, int axis, Tensor &out) = 0;

        protected:
            int m_axis = -1;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_STACK_H
