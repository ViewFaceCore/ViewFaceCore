//
// Created by kier on 2019/2/20.
//

#include <backend/base/base_activation.h>

#include "backend/base/base_activation.h"

namespace ts {
    namespace base {
        int Activation::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            output.resize(1);
            output[0] = stack[0].proto();

            return 1;
        }

        int Activation::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(x.proto(), memory_device);

            active(x, out);

            return 1;
        }
    }
}