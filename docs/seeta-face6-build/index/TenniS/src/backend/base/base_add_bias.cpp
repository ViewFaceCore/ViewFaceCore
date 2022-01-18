//
// Created by kier on 2019/2/15.
//

#include "backend/base/base_add_bias.h"

#include <utils/assert.h>
#include <numeric>
#include <backend/base/base_add_bias.h>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        AddBias::AddBias() {
            field(name::format, OPTIONAL);
            field(name::dim, OPTIONAL);
        }

        void AddBias::init() {
            supper::init();

            bool has_set_format = this->has(name::format);
            bool has_set_dim = this->has(name::dim);

            TS_AUTO_CHECK(has_set_format || has_set_dim);

            if (has_set_format) {
                m_format = tensor::to_string(this->get(name::format));

                TS_AUTO_CHECK(m_format == name::NCHW || m_format == name::NHWC);

                m_dim = int(m_format.find('C'));
            }

            if (has_set_dim) {
                m_dim = tensor::to_int(this->get(name::dim));
            }

            TS_AUTO_CHECK(m_dim >= 0);

        }

        bool AddBias::check_inputs(Stack &stack) const {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto &b = stack[1];

            TS_AUTO_CHECK(m_dim < x.dims());

            if(!b.has_shape(x.size(m_dim))) {
                TS_LOG_ERROR << "Miss matched: x:" << to_string(x.sizes()) <<
                             ", dim=" << m_dim <<
                             ", b:" << to_string(b.sizes()) << eject;
            }

            TS_AUTO_CHECK(x.dtype() == b.dtype());

            return true;
        }

        int AddBias::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            check_inputs(stack);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int AddBias::run(Stack &stack) {
            check_inputs(stack);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto b = stack[1].view(memory_device);

            auto out = *stack.push(x.proto(), memory_device);

            add(x, b, m_dim, out);

            return 1;
        }
    }
}
