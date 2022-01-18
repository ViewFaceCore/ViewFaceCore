//
// Created by kier on 2019/2/18.
//

#include <backend/base/base_pad.h>

#include "backend/base/base_pad.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Pad::Pad() {
            field(name::padding_value, OPTIONAL, tensor::from(0.0f));
        }

        void Pad::init() {
            supper::init();

            m_padding_value = tensor::to_float(get(name::padding_value));
        }

        int Pad::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto padding = tensor::cast(INT32, stack[1]);

            TS_AUTO_CHECK(padding.has_shape({-1, 2}));

            TS_AUTO_CHECK(padding.size(0) == x.dims());

            Shape output_shape = x.sizes();

            for (size_t i = 0; i < output_shape.size(); ++i) {
                output_shape[i] += (padding.data<int32_t>(i * 2) + padding.data<int32_t>(i * 2 + 1));
                if (output_shape[i] < 1) output_shape[i] = 1;
            }

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), output_shape);

            return 1;
        }

        int Pad::run(ts::Stack &stack) {
            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto padding_tensor = tensor::cast(INT32, stack[1]);

            bool is_zero_padding = true;
            int padding_tensor_count = padding_tensor.count();
            auto padding_tensor_data = padding_tensor.data<int32_t>();
            for (int i = 0; i < padding_tensor_count; ++i) {
                if (padding_tensor_data[i] != 0) {
                    is_zero_padding = false;
                    break;
                }
            }

            if (is_zero_padding) {
                stack.push(0);
                return 1;
            }

            auto &out = *stack.push(output[0], memory_device);

            std::vector<std::array<int, 2>> padding;

            for (int i = 0; i < out.dims(); ++i) {
                padding.emplace_back(std::array<int, 2>(
                        {padding_tensor.data<int32_t>(size_t(i * 2)),
                         padding_tensor.data<int32_t>(size_t(i * 2 + 1))}));
            }

            pad(x, padding, m_padding_value, out);

            return 1;
        }
    }
}
