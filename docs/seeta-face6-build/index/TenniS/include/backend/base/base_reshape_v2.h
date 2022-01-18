//
// Created by kier on 2019/3/5.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_RESHAPE_V2_H
#define TENSORSTACK_BACKEND_BASE_BASE_RESHAPE_V2_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class ReshapeV2 : public OperatorOnDevice {
        public:
            using self = ReshapeV2;
            using supper = OperatorOnDevice;

            ReshapeV2() = default;

            void init() override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param stack x, shape
             * @return
             */
            int run(Stack &stack) override;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_RESHAPE_V2_H
