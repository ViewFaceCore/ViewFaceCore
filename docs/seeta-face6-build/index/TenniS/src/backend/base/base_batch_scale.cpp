//
// Created by kier on 2019/2/15.
//

#include "backend/base/base_batch_scale.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        BatchScale::BatchScale() {
            field(name::dim, REQUIRED);
        }

        void BatchScale::init() {
            supper::init();

            m_dim = tensor::to_int(this->get(name::dim));

            TS_AUTO_CHECK(m_dim >= 0);

        }

        bool BatchScale::check_inputs(Stack &stack) const {
            TS_AUTO_CHECK(stack.size() == 3);

            auto &x = stack[0];
            auto &scale = stack[1];
            auto &bias = stack[2];

            TS_AUTO_CHECK(m_dim < x.dims());

            if(!scale.has_shape(x.size(m_dim))) {
                TS_LOG_ERROR << "Miss matched: x:" << to_string(x.sizes()) <<
                             ", dim=" << m_dim <<
                             ", scale:" << to_string(scale.sizes()) << eject;
            }

            if(!bias.has_shape(x.size(m_dim))) {
                TS_LOG_ERROR << "Miss matched: x:" << to_string(x.sizes()) <<
                             ", dim=" << m_dim <<
                             ", bias:" << to_string(bias.sizes()) << eject;
            }

            TS_AUTO_CHECK(x.dtype() == scale.dtype());
            TS_AUTO_CHECK(x.dtype() == bias.dtype());

            return true;
        }

        int BatchScale::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            check_inputs(stack);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int BatchScale::run(Stack &stack) {
            check_inputs(stack);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto scale = stack[1].view(memory_device);
            auto bias = stack[2].view(memory_device);

            auto out = *stack.push(x.proto(), memory_device);

            batch_scale(x, scale, bias, m_dim, out);

            return 1;
        }
    }
}
