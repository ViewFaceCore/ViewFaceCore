//
// Created by kier on 2019/3/6.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_GATHERV2_H
#define TENSORSTACK_BACKEND_BASE_BASE_GATHERV2_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class GatherV2 : public OperatorOnDevice {
        public:
            using self = GatherV2;
            using supper = OperatorOnDevice;

            GatherV2();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param indices int32 tensor on CPU
             * @param out
             */
            virtual void gather(const Tensor &x, const Tensor &indices, Tensor &out) = 0;

        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_GATHERV2_H
