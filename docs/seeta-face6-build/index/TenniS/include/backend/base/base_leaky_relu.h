//
// Created by kier on 2020/1/9.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_LEAKY_RELU_H
#define TENSORSTACK_BACKEND_BASE_BASE_LEAKY_RELU_H

#include "base_activation.h"

namespace ts {
    namespace base {
        class LeakyReLU : public Activation {
        public:
            LeakyReLU();

            void init() override;

            void active(const Tensor &x, Tensor &out) final;

            virtual void leaky_relu(const Tensor &x, float scale, Tensor &out) = 0;

        private:
            float m_scale = 0;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_LEAKY_RELU_H
