//
// Created by kier on 2019/2/18.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_NEW_SHAPE_H
#define TENSORSTACK_BACKEND_BASE_BASE_NEW_SHAPE_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        class NewShape : public OperatorOnDevice {
        public:
            using self = NewShape;
            using supper = OperatorOnDevice;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual Shape newshape(const Tensor &x) = 0;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_NEW_SHAPE_H
