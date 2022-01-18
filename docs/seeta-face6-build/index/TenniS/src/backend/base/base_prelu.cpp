//
// Created by kier on 2019/2/20.
//

#include "backend/base/base_prelu.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        PReLU::PReLU() {
            field(name::dim, REQUIRED);
        }

        void PReLU::init() {
            supper::init();

            m_dim = tensor::to_int(this->get(name::dim));

            TS_AUTO_CHECK(m_dim >= 0);
        }

        bool PReLU::check_inputs(Stack &stack) const {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto &slope = stack[1];

            TS_AUTO_CHECK(m_dim < x.dims());

            if(!slope.has_shape(x.size(m_dim))) {
                TS_LOG_ERROR << "Miss matched: x:" << to_string(x.sizes()) <<
                             ", dim=" << m_dim <<
                             ", slope:" << to_string(slope.sizes()) << eject;
            }

            TS_AUTO_CHECK(x.dtype() == slope.dtype());

            return true;
        }

        int PReLU::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            check_inputs(stack);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int PReLU::run(Stack &stack) {
            check_inputs(stack);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto slope = stack[1].view(memory_device);

            auto out = *stack.push(x.proto(), memory_device);

            prelu(x, slope, m_dim, out);

            return 1;
        }
    }
}
