//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_ADD_BIAS_H
#define TENSORSTACK_BACKEND_BASE_ADD_BIAS_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         * AddBias
         */
        class AddBias : public OperatorOnDevice {
        public:
            using self = AddBias;
            using supper = OperatorOnDevice;

            AddBias();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void add(const Tensor &x, const Tensor &b, int dim, Tensor &out) = 0;

        private:
            std::string m_format;
            int m_dim = -1;

            bool check_inputs(Stack &stack) const;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_ADD_BIAS_H
