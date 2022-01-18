//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_SOFTMAX_H
#define TENSORSTACK_BACKEND_BASE_SOFTMAX_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         * PReLU
         */
        class Softmax : public OperatorOnDevice {
        public:
            using self = Softmax;
            using supper = OperatorOnDevice;

            Softmax();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void softmax(const Tensor &x, int dim, bool smooth, Tensor &out) = 0;

        private:
            int m_dim = -1;
            bool m_smooth = true;

            bool check_inputs(Stack &stack) const;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_SOFTMAX_H
