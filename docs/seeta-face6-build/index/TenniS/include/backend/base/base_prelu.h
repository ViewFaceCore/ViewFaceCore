//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_PRELU_H
#define TENSORSTACK_BACKEND_BASE_PRELU_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         * PReLU
         */
        class PReLU : public OperatorOnDevice {
        public:
            using self = PReLU;
            using supper = OperatorOnDevice;

            PReLU();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void prelu(const Tensor &x, const Tensor &slope, int dim, Tensor &out) = 0;

        private:
            int m_dim = -1;

            bool check_inputs(Stack &stack) const;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_PRELU_H
