#ifndef TENSORSTACK_BACKEND_BASE_BASE_MAX_H
#define TENSORSTACK_BACKEND_BASE_BASE_MAX_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Max : public OperatorOnDevice {
        public:
            using self = Max;
            using supper = OperatorOnDevice;

            Max();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param out
             */
            virtual void max(const Tensor &x, Tensor &out) = 0;

        protected:
            int m_dim;
            int m_keep_dims;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_MAX_H
