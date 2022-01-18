//
// Created by kier on 2019/2/19.
//

#include "backend/base/base_global_pooling2d.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

#include "backend/common_function.h"
#include "utils/need.h"

namespace ts {
    namespace base {
        GlobalPooling2D::GlobalPooling2D() {
            field(name::format, REQUIRED);
            // field(name::type, OPTIONAL, tensor::from(int(Pooling2DType::MAX)));  // use optional running failed
            field(name::type, REQUIRED);
        }

        void GlobalPooling2D::init() {
            supper::init();

            auto format = tensor::to_string(get(name::format));
            m_type = static_cast<Pooling2DType>(tensor::to_int(get(name::type)));

            if (format == name::NCHW) {
                m_format = FORMAT_NCHW;
            } else if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            } else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }
        }

        int GlobalPooling2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x_tensor = stack[0];

            TS_AUTO_CHECK(x_tensor.dims() == 4);

            if (m_format == FORMAT_NCHW) {
                output.resize(1);
                output[0] = Tensor::Prototype(
                        x_tensor.dtype(),
                        {x_tensor.size(0), x_tensor.size(1), 1, 1});
            } else if (m_format == FORMAT_NHWC) {
                output.resize(1);
                output[0] = Tensor::Prototype(
                        x_tensor.dtype(),
                        {x_tensor.size(0), 1, 1, x_tensor.size(3)});
            }

            return 1;
        }

        int GlobalPooling2D::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            Tensor x = stack[0].view(memory_device);

            Tensor out = *stack.push(output_protos[0], memory_device);

            auto &x_shape = x.sizes();
            Size2D ksize;
            static const Padding2D padding(0, 0, 0, 0);
            static const Stride2D stride(1, 1);

            if (m_format == FORMAT_NCHW) {
                ksize = Size2D(x_shape[2], x_shape[3]);
            } else if (m_format == FORMAT_NHWC) {
                ksize = Size2D(x_shape[1], x_shape[2]);
            }

            pooling2d(x, m_type, padding, Padding2DType::BLACK, ksize, stride, m_format, out);

            return 1;
        }
    }
}

