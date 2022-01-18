//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_CONCAT_H
#define TENSORSTACK_BACKEND_BASE_BASE_CONCAT_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Concat : public OperatorOnDevice {
        public:
            using self = Concat;
            using supper = OperatorOnDevice;

            Concat();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void concat(const std::vector<Tensor> &x, int dim, Tensor &out) = 0;

        private:
            int m_dim = -1;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_CONCAT_H
