//
// Created by kier on 2019/1/12.
//

#include <backend/mxnet/pooling2d_padding.h>

#include "backend/mxnet/pooling2d_padding.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "utils/assert.h"

#include "runtime/stack.h"

#include "backend/common_function.h"

#include "core/device.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"

namespace ts {
    namespace mxnet {

        Pooling2dPadding::Pooling2dPadding() {
            this->field(name::valid, REQUIRED);
            this->field(name::format, REQUIRED);
            this->field(name::padding, REQUIRED);
        }

        void Pooling2dPadding::init() {
            supper::init();

            this->valid = tensor::to_int(this->get(name::valid)) != 0;
            this->format = tensor::to_string(this->get(name::format));

            TS_AUTO_CHECK(format == name::NCHW || format == name::NHWC);

            auto static_padding = tensor::cast(INT32, this->get(name::padding));

            TS_AUTO_CHECK(static_padding.has_shape({4, 2}));

            if (format == name::NCHW) {
                this->static_padding.top = static_padding.data<int32_t>()[4];
                this->static_padding.bottom = static_padding.data<int32_t>()[5];
                this->static_padding.left = static_padding.data<int32_t>()[6];
                this->static_padding.right = static_padding.data<int32_t>()[7];
            } else {
                this->static_padding.top = static_padding.data<int32_t>()[2];
                this->static_padding.bottom = static_padding.data<int32_t>()[3];
                this->static_padding.left = static_padding.data<int32_t>()[4];
                this->static_padding.right = static_padding.data<int32_t>()[5];
            }
        }

        int Pooling2dPadding::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            output.resize(1);
            output[0] = Tensor::Prototype(INT32, {4, 2});
            return 1;
        }

        static Size2D valid_pooling2d_forward(const Size2D &x, const Padding2D &padding, const KSize2D &ksize,
                                        const Stride2D &stride) {
            Size2D y;
            y.height = int(std::floor((x.height + padding.top + padding.bottom - ksize.height) / (float)stride.height + 1));
            y.width = int(std::floor((x.width + padding.left + padding.right - ksize.width) / (float)stride.width + 1));
            return y;
        }

        int Pooling2dPadding::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            // get output
            auto &dynamic_padding_tensor = *stack.push(INT32, {4, 2}, MemoryDevice(CPU));

            // convert input
            auto shape = stack.index(0)->sizes();
            auto ksize_tensor = tensor::cast(INT32, stack.index(1)->view(MemoryDevice(CPU)));
            auto stride_tensor = tensor::cast(INT32, stack.index(2)->view(MemoryDevice(CPU)));

            Padding2D dynamic_padding;

            if (this->valid) {
                KSize2D ksize;
                Stride2D stride;
                Size2D input_size;

                if (format == name::NCHW) {
                    ksize.height = ksize_tensor.data<int32_t>()[2];
                    ksize.width = ksize_tensor.data<int32_t>()[3];
                    stride.height = stride_tensor.data<int32_t>()[2];
                    stride.width = stride_tensor.data<int32_t>()[3];
                    input_size.height = shape[2];
                    input_size.width = shape[3];
                } else {
                    ksize.height = ksize_tensor.data<int32_t>()[1];
                    ksize.width = ksize_tensor.data<int32_t>()[2];
                    stride.height = stride_tensor.data<int32_t>()[1];
                    stride.width = stride_tensor.data<int32_t>()[2];
                    input_size.height = shape[1];
                    input_size.width = shape[2];
                }

                auto t = (valid_pooling2d_forward);
				TS_UNUSED(t);

                /* Heavy version
                Size2D expected_output_size = valid_pooling2d_forward(input_size, static_padding, ksize, stride);
                Size2D expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride);
                dynamic_padding.top = static_padding.top;
                dynamic_padding.left = static_padding.left;
                dynamic_padding.bottom = static_padding.bottom + expected_input_size.height - input_size.height;
                dynamic_padding.right = static_padding.right + expected_input_size.width - input_size.width;
                 */
                /* light version */
                dynamic_padding.top = static_padding.top;
                dynamic_padding.left = static_padding.left;
                dynamic_padding.bottom = static_padding.bottom
                                         - (input_size.height + static_padding.top + static_padding.bottom - ksize.height) % stride.height;
                dynamic_padding.right = static_padding.right
                                        - (input_size.width + static_padding.left + static_padding.right - ksize.width) % stride.width;
            } else {
                dynamic_padding = static_padding;
            }

            auto dynamic_padding_data = dynamic_padding_tensor.data<int32_t>();

            if (format == name::NCHW) {
                dynamic_padding_data[0] = 0;
                dynamic_padding_data[1] = 0;
                dynamic_padding_data[2] = 0;
                dynamic_padding_data[3] = 0;
                dynamic_padding_data[4] = dynamic_padding.top;
                dynamic_padding_data[5] = dynamic_padding.bottom;
                dynamic_padding_data[6] = dynamic_padding.left;
                dynamic_padding_data[7] = dynamic_padding.right;

            } else {
                dynamic_padding_data[0] = 0;
                dynamic_padding_data[1] = 0;
                dynamic_padding_data[6] = 0;
                dynamic_padding_data[7] = 0;
                dynamic_padding_data[2] = dynamic_padding.top;
                dynamic_padding_data[3] = dynamic_padding.bottom;
                dynamic_padding_data[4] = dynamic_padding.left;
                dynamic_padding_data[5] = dynamic_padding.right;
            }

            return 1;
        }
    }
}

using namespace ts;
using namespace ts::mxnet;

TS_REGISTER_OPERATOR(Pooling2dPadding, CPU, name::layer::mx_pooling2d_padding())

TS_REGISTER_FP16_OPERATOR(Pooling2dPadding, GPU, name::layer::mx_pooling2d_padding())
