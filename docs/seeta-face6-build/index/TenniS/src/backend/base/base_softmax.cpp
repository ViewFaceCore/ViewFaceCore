//
// Created by kier on 2019/2/20.
//

#include "backend/base/base_softmax.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        Softmax::Softmax() {
            field(name::dim, REQUIRED);
            field(name::smooth, OPTIONAL, tensor::from<bool>(true));
        }

        void Softmax::init() {
            supper::init();

            m_dim = tensor::to_int(this->get(name::dim));
            m_smooth = tensor::to_bool(get(name::smooth));

            // TS_AUTO_CHECK(m_dim >= 0);
        }

        bool Softmax::check_inputs(Stack &stack) const {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            int output_dims = int(x.dims());
            int fixed_dim = m_dim >= 0 ? m_dim : output_dims + m_dim;

            if (fixed_dim < 0 || fixed_dim >= output_dims) {
                TS_LOG_ERROR << "Softmax dim must in [-"
                             << output_dims << ", "
                             << output_dims << ")" << eject;
            }

            return true;
        }

        int Softmax::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            check_inputs(stack);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int Softmax::run(Stack &stack) {
            check_inputs(stack);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto out = *stack.push(x.proto(), memory_device);

            int output_dims = int(x.dims());
            int fixed_dim = m_dim >= 0 ? m_dim : output_dims + m_dim;

            softmax(x, fixed_dim, m_smooth, out);

            return 1;
        }
    }
}
