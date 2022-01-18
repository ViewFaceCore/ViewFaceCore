//
// Created by kier on 2019/6/26.
//

#include "backend/base/base_force_color.h"

namespace ts {
    namespace base {
        ForceColor::ForceColor() {
        }

        void ForceColor::init() {
            supper::init();
        }

        int ForceColor::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() > 0);

            output.resize(1);
            auto output_size = x.sizes();
            output_size.back() = 3;

            output[0] = Tensor::Prototype(x.dtype(), std::move(output_size));

            return 1;
        }

        int ForceColor::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() > 0);

            auto channels = x.size(x.dims() - 1);

            if (channels == 3) return 1;

            if (channels != 1) {
                TS_LOG_ERROR << "Can not force image " << to_string(x.sizes()) << " to color." << eject;
            }

            auto output_size = x.sizes();
            output_size.back() = 3;

            auto memory_device = running_memory_device();

            auto image = x.view(memory_device);
            auto &out = *stack.push(image.dtype(), output_size, memory_device);

            force_color(image, out);

            return 1;
        }


    }
}