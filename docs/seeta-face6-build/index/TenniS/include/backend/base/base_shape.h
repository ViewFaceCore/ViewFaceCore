//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_SHAPE_H
#define TENSORSTACK_BACKEND_BASE_BASE_SHAPE_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class GetShape : public OperatorOnDevice {
        public:
            using self = GetShape;
            using supper = OperatorOnDevice;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_SHAPE_H
