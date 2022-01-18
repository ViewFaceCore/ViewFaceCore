#include "backend/base/base_quantize.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        Quantize::Quantize() {
            field(name::quantize_scale, REQUIRED);
        }

        void Quantize::init() {
            supper::init();

            m_quantize_scales = tensor::array::to_float(this->get(name::quantize_scale));
        }

        int Quantize::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x_tensor = stack[0];
            TS_AUTO_CHECK(m_quantize_scales.size() == x_tensor.sizes()[0]);

            output.resize(1);
            auto input_proto = stack[0].proto();
            Tensor::Prototype output_proto(INT8, input_proto.sizes());
            output[0] = output_proto;

            return 1;
        }

        int Quantize::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto out = *stack.push(output_protos[0], memory_device);

            quantize(x, m_quantize_scales, out);

            return 1;
        }
    }
}