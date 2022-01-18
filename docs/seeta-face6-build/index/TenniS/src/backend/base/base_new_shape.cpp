//
// Created by kier on 2019/2/18.
//

#include <backend/base/base_new_shape.h>

#include "backend/base/base_new_shape.h"

namespace ts {
    namespace base {

        void NewShape::init() {
            supper::init();
        }

        int NewShape::run(Stack &stack) {
            std::vector<Tensor::Prototype> output;

            infer(stack, output);


            auto &x = stack[0];

            stack.push(x.reshape(output[0].sizes()));

            return 1;
        }

        int NewShape::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto input_num = stack.size();
            TS_AUTO_CHECK(input_num == 1);

            auto &x = stack[0];

            auto new_shape = this->newshape(x);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), new_shape);

            auto &reshape_x = output[0];

            if (x.count() != reshape_x.count()) {
                TS_LOG_ERROR << "Can not reshape " << to_string(x.sizes()) << " to " << to_string(new_shape) << eject;
            }

            return 1;
        }
    }
}