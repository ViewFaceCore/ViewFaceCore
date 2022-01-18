//
// Created by kier on 2019/2/18.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_BROADCAST_H
#define TENSORSTACK_BACKEND_BASE_BASE_BROADCAST_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Broadcast : public OperatorOnDevice {
        public:
            using self = Broadcast;
            using supper = OperatorOnDevice;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            int run(Stack &stack) override;

            virtual void broadcast(const Tensor &x, const std::vector<int32_t> &shape, Tensor &out) = 0;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_BROADCAST_H
