//
// Created by kier on 2019/2/19.
//

#include "backend/base/base_pooling2d.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

#include "backend/common_function.h"
#include "utils/need.h"

namespace ts {
    namespace base {
        Pooling2D::Pooling2D() {
            field(name::format, REQUIRED);
            // field(name::type, OPTIONAL, tensor::from(int(Pooling2DType::MAX)));  // use optional running failed
            field(name::type, REQUIRED);
            field(name::padding, REQUIRED);
            field(name::padding_type, OPTIONAL, tensor::from(int(Padding2DType::BLACK)));
            field(name::ksize, REQUIRED);
            field(name::stride, REQUIRED);
        }

        static std::string to_string(const std::valarray<int> &arr) {
            std::ostringstream out;
            out << "[";
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i) out << ", ";
                out << arr[i];
            }
            out << "]";
            return out.str();
        }


        void Pooling2D::init() {
            supper::init();

            auto format = tensor::to_string(get(name::format));
            m_type = static_cast<Pooling2DType>(tensor::to_int(get(name::type)));
            auto padding_tensor = tensor::cast(INT32, get(name::padding));
            m_padding_type = static_cast<Padding2DType>(tensor::to_int(get(name::padding_type)));
            auto ksize_tensor = tensor::cast(INT32, get(name::ksize));
            auto stride_tensor = tensor::cast(INT32, get(name::stride));

            TS_AUTO_CHECK(padding_tensor.has_shape({4, 2}));
            TS_AUTO_CHECK(ksize_tensor.has_shape({4,}));
            TS_AUTO_CHECK(stride_tensor.has_shape({4,}));

            if (format == name::NCHW) {
                m_format = FORMAT_NCHW;
            } else if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            } else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }

            m_padding4x2.resize(8);
            for (size_t i = 0; i < 8; ++i) m_padding4x2[i] = padding_tensor.data<int32_t>(i);
            m_ksize4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_ksize4[i] = ksize_tensor.data<int32_t>(i);
            m_stride4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_stride4[i] = stride_tensor.data<int32_t>(i);

            // only support native conv2d
            if (m_format == FORMAT_NCHW) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[2] != 0 ||
                    m_padding4x2[3] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
                if (m_ksize4[0] != 1 ||
                    m_ksize4[1] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support ksize: " << to_string(m_ksize4) << eject;
                }
                if (m_stride4[0] != 1 ||
                    m_stride4[1] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support stride: " << to_string(m_stride4) << eject;
                }
            } else if (m_format == FORMAT_NHWC) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[6] != 0 ||
                    m_padding4x2[7] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
                if (m_ksize4[0] != 1 ||
                    m_ksize4[3] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support ksize: " << to_string(m_ksize4) << eject;
                }
                if (m_stride4[0] != 1 ||
                    m_stride4[3] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support stride: " << to_string(m_stride4) << eject;
                }
            }
        }

        int Pooling2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x_tensor = stack[0];

            TS_AUTO_CHECK(x_tensor.dims() == 4);

            Size2D x;
            Size2D ksize;
            Padding2D padding;
            Stride2D stride;

            if (m_format == FORMAT_NCHW) {
                x = Size2D(x_tensor.size(2), x_tensor.size(3));
                ksize = Size2D(m_ksize4[2], m_ksize4[3]);
                padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
                stride = Stride2D(m_stride4[2], m_stride4[3]);
            } else if (m_format == FORMAT_NHWC) {
                x = Size2D(x_tensor.size(1), x_tensor.size(2));
                ksize = Size2D(m_ksize4[1], m_ksize4[2]);
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
                stride = Stride2D(m_stride4[1], m_stride4[2]);
            }

            Size2D y = pooling2d_forward(x, padding, ksize, stride);

            Tensor::Prototype out_proto;

            if (m_format == FORMAT_NCHW) {
                out_proto = Tensor::Prototype(
                        x_tensor.dtype(),
                        {x_tensor.size(0), x_tensor.size(1), y.height, y.width});
            } else if (m_format == FORMAT_NHWC) {
                out_proto = Tensor::Prototype(
                        x_tensor.dtype(),
                        {x_tensor.size(0), y.height, y.width, x_tensor.size(3)});
            }

            output.resize(1);
            output[0] = out_proto;

            return 1;
        }

        int Pooling2D::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            Tensor x = stack[0].view(memory_device);

            Tensor out = *stack.push(output_protos[0], memory_device);

            Size2D ksize;
            Padding2D padding;
            Stride2D stride;

            if (m_format == FORMAT_NCHW) {
                ksize = Size2D(m_ksize4[2], m_ksize4[3]);
                padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
                stride = Stride2D(m_stride4[2], m_stride4[3]);
            } else if (m_format == FORMAT_NHWC) {
                ksize = Size2D(m_ksize4[1], m_ksize4[2]);
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
                stride = Stride2D(m_stride4[1], m_stride4[2]);
            }

            pooling2d(x, m_type, padding, m_padding_type, ksize, stride, m_format, out);

            return 1;
        }
    }
}

