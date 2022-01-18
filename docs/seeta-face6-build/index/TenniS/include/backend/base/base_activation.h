//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_ACTIVATION_H
#define TENSORSTACK_BACKEND_BASE_BASE_ACTIVATION_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Activation : public OperatorOnDevice {
        public:
            using self = Activation;
            using supper = OperatorOnDevice;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void active(const Tensor &x, Tensor &out) = 0;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_ACTIVATION_H
