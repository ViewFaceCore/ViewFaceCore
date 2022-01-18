#ifndef TENSORSTACK_BACKEND_BASE_BASE_SLICE_V2_H
#define TENSORSTACK_BACKEND_BASE_BASE_SLICE_V2_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class SliceV2 : public OperatorOnDevice {
        public:
            using self = SliceV2;
            using supper = OperatorOnDevice;

            SliceV2();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param out
             */
            virtual void slice(const Tensor &x, const std::vector<int> &begins, const std::vector<int> & sizes, Tensor &out) = 0;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_SLICE_V2_H
