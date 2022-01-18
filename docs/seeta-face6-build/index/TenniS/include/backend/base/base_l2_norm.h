//
// Created by kier on 2019/6/12.
//

#ifndef TENSORSTACK_BACKEND_BASE_L2_NORM_H
#define TENSORSTACK_BACKEND_BASE_L2_NORM_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         *
         */
        class L2Norm : public OperatorOnDevice {
        public:
            using self = L2Norm;
            using supper = OperatorOnDevice;

            L2Norm();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void normalize(const Tensor &x, int dim, float epsilon, Tensor &out) = 0;

        private:
            int m_dim = -1;
            float m_epsilon = 1.00000001e-10f;

            bool check_inputs(Stack &stack) const;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_L2_NORM_H
