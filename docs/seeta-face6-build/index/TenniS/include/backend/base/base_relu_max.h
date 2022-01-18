//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_RELU_MAX_H
#define TENSORSTACK_BACKEND_BASE_BASE_RELU_MAX_H

#include "base_activation.h"

namespace ts {
    namespace base {
        class ReLUMax : public Activation {
        public:
            ReLUMax();

            void init() override;

            void active(const Tensor &x, Tensor &out) final;

            virtual void relu_max(const Tensor &x, float max, Tensor &out) = 0;

        private:
            float m_max;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_RELU_MAX_H
