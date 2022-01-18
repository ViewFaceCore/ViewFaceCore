//
// Created by kier on 2019/6/12.
//

#include "backend/base/base_l2_norm.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        L2Norm::L2Norm() {
            field(name::dim, OPTIONAL, tensor::from<int32_t>(-1));
            field(name::epsilon, OPTIONAL, tensor::from<float>(m_epsilon));
        }

        void L2Norm::init() {
            supper::init();

            m_dim = tensor::to_int(this->get(name::dim));
            m_epsilon = tensor::to_float(this->get(name::epsilon));

            // TS_AUTO_CHECK(m_dim >= 0);
        }

        bool L2Norm::check_inputs(Stack &stack) const {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            int output_dims = int(x.dims());
            int fixed_dim = m_dim >= 0 ? m_dim : output_dims + m_dim;

            if (fixed_dim < 0 || fixed_dim >= output_dims) {
                TS_LOG_ERROR << "L2Norm dim must in [-"
                             << output_dims << ", "
                             << output_dims << ")" << eject;
            }

            return true;
        }

        int L2Norm::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            check_inputs(stack);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int L2Norm::run(Stack &stack) {
            check_inputs(stack);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto out = *stack.push(x.proto(), memory_device);

            int output_dims = int(x.dims());
            int fixed_dim = m_dim >= 0 ? m_dim : output_dims + m_dim;

            normalize(x, fixed_dim, m_epsilon, out);

            return 1;
        }
    }
}
