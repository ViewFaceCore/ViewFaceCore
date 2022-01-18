//
// Created by kier on 2019/2/20.
//

#include "backend/base/base_shape.h"

namespace ts {
    namespace base {
        int GetShape::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            output.resize(1);
            output[0] = Tensor::Prototype(INT32, {int(x.dims())});

            return 1;
        }

        int GetShape::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            auto dims = int(x.dims());
            auto shape = *stack.push(INT32, {dims,}, MemoryDevice(CPU));
            auto shape_data = shape.data<int32_t>();

            for (int i = 0; i < dims; ++i) {
                shape_data[i] = x.size(i);
            }

            return 1;
        }
    }
}
