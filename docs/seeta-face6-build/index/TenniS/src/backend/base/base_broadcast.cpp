//
// Created by kier on 2019/10/17.
//

#include "backend/base/base_broadcast.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        int Broadcast::infer(ts::Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto &shape = stack[1];

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), tensor::array::to_int(shape));

            return 1;
        }

        static bool can_broadcast(const Shape &x, const Shape &y) {
            if (x.size() != y.size()) return false;
            auto N = x.size();
            for (size_t i = 0; i < N; ++i) {
                if (x[i] != 1 && x[i] != y[i]) return false;
            }
            return true;
        }

        int Broadcast::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto shape = tensor::array::to_int(stack[1]);

            if (!can_broadcast(x.sizes(), shape)) {
                TS_LOG_ERROR << "Can not broadcast x.shape="
                             << to_string(x.sizes()) << " to " << to_string(shape) << eject;
            }

            auto &out = *stack.push(x.dtype(), shape, memory_device);

            broadcast(x, shape, out);

            return 1;
        }
    }
}
