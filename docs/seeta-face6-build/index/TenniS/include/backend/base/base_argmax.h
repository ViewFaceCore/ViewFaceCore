#ifndef TENSORSTACK_BACKEND_BASE_BASE_ARGMAX_H
#define TENSORSTACK_BACKEND_BASE_BASE_ARGMAX_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class ArgMax : public OperatorOnDevice {
        public:
            using self = ArgMax;
            using supper = OperatorOnDevice;

            ArgMax();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param out
             */
            virtual void argmax(const Tensor &x, int dim, Tensor &out) = 0;

        protected:
            int m_dim;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_ARGMAX_H
