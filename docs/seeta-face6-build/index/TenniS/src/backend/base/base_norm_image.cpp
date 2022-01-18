//
// Created by kier on 2019/6/29.
//

#include <backend/base/base_norm_image.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        NormImage::NormImage() {
            field(name::epsilon, OPTIONAL, tensor::from<float>(1e-5f));
        }

        void NormImage::init() {
            supper::init();
            m_epsilon = tensor::to_float(get(name::epsilon));
        }

        int NormImage::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int NormImage::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(x.proto(), memory_device);

            norm_image(x, m_epsilon, out);

            return 1;
        }
    }
}