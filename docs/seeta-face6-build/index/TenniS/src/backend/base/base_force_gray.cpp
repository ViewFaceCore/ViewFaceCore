//
// Created by kier on 2019/6/26.
//

#include "backend/base/base_force_gray.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        ForceGray::ForceGray() {
            field("scale", OPTIONAL);
        }

        void ForceGray::init() {
            supper::init();

            m_scale.clear();
            if (has("scale")) {
                m_scale = tensor::array::to_float(get("scale"));
            }
        }

        int ForceGray::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() > 0);

            output.resize(1);
            auto output_size = x.sizes();
            output_size.back() = 1;

            output[0] = Tensor::Prototype(x.dtype(), std::move(output_size));

            return 1;
        }

        int ForceGray::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() > 0);

            auto channels = x.size(x.dims() - 1);

            if (channels == 1) return 1;

            auto output_size = x.sizes();
            output_size.back() = 1;

            auto memory_device = running_memory_device();

            auto image = x.view(memory_device);
            auto &out = *stack.push(image.dtype(), output_size, memory_device);

            auto scale = m_scale;
            if (scale.empty()) {
                if (channels != 3) {
                    TS_LOG_ERROR << "Can not force image " << to_string(x.sizes()) << " to gray." << eject;
                }
                static const std::vector<float> bgr_scale = {0.114f, 0.587f, 0.299f};
                force_gray(image, bgr_scale, out);
            } else {
                if (channels != m_scale.size()) {
                    TS_LOG_ERROR << "Can not force image " << to_string(x.sizes()) << " to gray." << eject;
                }
                force_gray(image, m_scale, out);
            }

            return 1;
        }
    }
}