//
// Created by kier on 2019/2/15.
//

#include "backend/base/base_fused_batch_norm.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        FusedBatchNorm::FusedBatchNorm() {
            field(name::epsilon, OPTIONAL);
            field(name::dim, REQUIRED);
        }

        void FusedBatchNorm::init() {
            supper::init();

            if (this->has(name::epsilon)) {
                m_epsilon = tensor::to_float(this->get(name::epsilon));
            }

            m_dim = tensor::to_int(this->get(name::dim));

            TS_AUTO_CHECK(m_dim >= 0);

        }

        bool FusedBatchNorm::check_inputs(Stack &stack) const {
            TS_AUTO_CHECK(stack.size() == 5);

            auto &x = stack[0];
            auto &mean = stack[1];
            auto &variance = stack[2];
            auto &scale = stack[3];
            auto &bias = stack[4];

            if (!(m_dim < x.dims()) ||
                !(mean.has_shape({x.size(m_dim)})) ||
                !(variance.has_shape({x.size(m_dim)})) ||
                !(scale.has_shape({x.size(m_dim)})) ||
                !(bias.has_shape({x.size(m_dim)})) ||
                !(x.dtype() == mean.dtype()) ||
                !(x.dtype() == variance.dtype()) ||
                !(x.dtype() == scale.dtype()) ||
                !(x.dtype() == bias.dtype())) {
                TS_LOG_ERROR << "FusedBatchNorm failed: x=" << x.proto()
                             << ", mean=" << mean.proto()
                             << ", variance=" << mean.proto()
                             << ", scale=" << mean.proto()
                             << ", bias=" << mean.proto()
                             << ", dim=" << m_dim
                             << "." << eject;
            }

            return true;
        }

        int FusedBatchNorm::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            check_inputs(stack);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int FusedBatchNorm::run(Stack &stack) {
            check_inputs(stack);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto mean = stack[1].view(memory_device);
            auto variance = stack[2].view(memory_device);
            auto scale = stack[3].view(memory_device);
            auto bias = stack[4].view(memory_device);

            auto out = *stack.push(x.proto(), memory_device);

            batch_norm(x, mean, variance, scale, bias, m_dim, m_epsilon, out);

            return 1;
        }
    }
}
