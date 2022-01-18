//
// Created by kier on 2020/1/9.
//

#include "backend/base/base_leaky_relu.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        LeakyReLU::LeakyReLU() {
            field(name::scale, OPTIONAL, tensor::from(float(0)));
        }

        void LeakyReLU::init() {
            supper::init();

            m_scale = tensor::to_float(get(name::scale));
        }

        void LeakyReLU::active(const Tensor &x, Tensor &out) {
            leaky_relu(x, m_scale, out);
        }
    }
}

