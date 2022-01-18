#ifndef TENSORSTACK_BACKEND_BASE_BASE_SLICE_H
#define TENSORSTACK_BACKEND_BASE_BASE_SLICE_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Slice : public OperatorOnDevice {
        public:
            using self = Slice;
            using supper = OperatorOnDevice;

            Slice();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param out
             */
            virtual void slice(const Tensor &x, Tensor &out) = 0;

        protected:
            std::vector<int> m_begin;
            std::vector<int> m_size;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_SLICE_H
