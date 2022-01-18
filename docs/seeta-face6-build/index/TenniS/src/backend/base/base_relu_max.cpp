//
// Created by kier on 2019/2/20.
//

#include <backend/base/base_relu_max.h>

#include "backend/base/base_relu_max.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        ReLUMax::ReLUMax() {
            field(name::max, OPTIONAL, tensor::from(float(0)));
        }

        void ReLUMax::init() {
            supper::init();

            m_max = tensor::to_float(get(name::max));
        }

        void ReLUMax::active(const Tensor &x, Tensor &out) {
            relu_max(x, m_max, out);
        }
    }
}

