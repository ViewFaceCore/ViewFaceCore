//
// Created by kier on 19-4-17.
//

#include "backend/torch/base_dcn_v2_forward.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

#include "backend/common_function.h"
#include "utils/need.h"

namespace ts {
    namespace base {
        DCNV2Forward::DCNV2Forward() {
            field(name::format, REQUIRED);
            field(name::padding, REQUIRED);
            field(name::deformable_groups, REQUIRED);
            field(name::stride, REQUIRED);
            field(name::dilation, REQUIRED);
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


        void DCNV2Forward::init() {
            supper::init();

            auto format = tensor::to_string(get(name::format));
            auto padding_tensor = tensor::cast(INT32, get(name::padding));
            m_deformable_groups = tensor::to_int(get(name::deformable_groups));
            auto stride_tensor = tensor::cast(INT32, get(name::stride));

            Tensor dilation_tensor = tensor::cast(INT32, get(name::dilation));

            if (dilation_tensor.empty()) {
                TS_LOG_ERROR << this->op() << " must set " << name::dilation << eject;
            }

            TS_AUTO_CHECK(padding_tensor.has_shape({4, 2}));
            TS_AUTO_CHECK(stride_tensor.has_shape({4,}));
            TS_AUTO_CHECK(dilation_tensor.has_shape({4,}));

            if (format == name::NCHW) {
                m_format = FORMAT_NCHW;
            } else if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            } else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }

            m_padding4x2.resize(8);
            for (size_t i = 0; i < 8; ++i) m_padding4x2[i] = padding_tensor.data<int32_t>(i);
            m_stride4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_stride4[i] = stride_tensor.data<int32_t>(i);
            m_dilation4.resize(4);
            for (size_t i = 0; i < 4; ++i) m_dilation4[i] = dilation_tensor.data<int32_t>(i);

            // only support native conv2d
            if (m_format == FORMAT_NCHW) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[2] != 0 ||
                    m_padding4x2[3] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
                if (m_stride4[0] != 1 ||
                    m_stride4[1] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support stride: " << to_string(m_stride4) << eject;
                }
                if (m_dilation4[0] != 1 ||
                    m_dilation4[1] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support dialations: " << to_string(m_dilation4) << eject;
                }
            } else if (m_format == FORMAT_NHWC) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[6] != 0 ||
                    m_padding4x2[7] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
                if (m_stride4[0] != 1 ||
                    m_stride4[3] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support stride: " << to_string(m_stride4) << eject;
                }
                if (m_dilation4[0] != 1 ||
                    m_dilation4[3] != 1) {
                    TS_LOG_ERROR << this->op() << " do not support dialations: " << to_string(m_dilation4) << eject;
                }
            }
        }

        int DCNV2Forward::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 5);

            auto x_tensor = stack[0];
            auto w_tensor = stack[1];
            auto b_tensor = stack[2];
            auto offset_tensor = stack[3];
            auto mask_tensor = stack[4];

            TS_AUTO_CHECK(x_tensor.dims() == 4);
            TS_AUTO_CHECK(w_tensor.dims() == 4);
            TS_AUTO_CHECK(b_tensor.dims() == 1);
            TS_AUTO_CHECK(offset_tensor.dims() == 4);
            TS_AUTO_CHECK(mask_tensor.dims() == 4);

            TS_AUTO_CHECK(x_tensor.dtype() == w_tensor.dtype());
            TS_AUTO_CHECK(x_tensor.dtype() == b_tensor.dtype());
            TS_AUTO_CHECK(x_tensor.dtype() == offset_tensor.dtype());
            TS_AUTO_CHECK(x_tensor.dtype() == mask_tensor.dtype());

            // TS_AUTO_CHECK(w_tensor.size(1) == x_tensor.size(1));

            Size2D x;
            Size2D ksize;
            Padding2D padding;
            Stride2D stride;
            Dilation2D dialations;

            if (m_format == FORMAT_NCHW) {
                x = Size2D(x_tensor.size(2), x_tensor.size(3));
                ksize = Size2D(w_tensor.size(2), w_tensor.size(3));
                padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
                stride = Stride2D(m_stride4[2], m_stride4[3]);
                dialations = Stride2D(m_dilation4[2], m_dilation4[3]);
            } else if (m_format == FORMAT_NHWC) {
                x = Size2D(x_tensor.size(1), x_tensor.size(2));
                ksize = Size2D(w_tensor.size(1), w_tensor.size(2));
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
                stride = Stride2D(m_stride4[1], m_stride4[2]);
                dialations = Stride2D(m_dilation4[1], m_dilation4[2]);
            }

            Size2D y = conv2d_forward(x, padding, ksize, stride, dialations);

            Tensor::Prototype out_proto;

            if (m_format == FORMAT_NCHW) {
                out_proto = Tensor::Prototype(
                        x_tensor.dtype(),
                        {x_tensor.size(0), w_tensor.size(0), y.height, y.width});
            } else if (m_format == FORMAT_NHWC) {
                out_proto = Tensor::Prototype(
                        x_tensor.dtype(),
                        {x_tensor.size(0), y.height, y.width, w_tensor.size(0)});
            }

            output.resize(1);
            output[0] = out_proto;

            return 1;
        }

        int DCNV2Forward::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            Tensor x = stack[0].view(memory_device);
            Tensor w = stack[1].view(memory_device);
            Tensor b = stack[2].view(memory_device);
            Tensor offset = stack[3].view(memory_device);
            Tensor mask = stack[4].view(memory_device);

            Tensor out = *stack.push(output_protos[0], memory_device);

            Padding2D padding;
            Stride2D stride;
            Dilation2D dilation;
            // Size2D ksize;

            if (m_format == FORMAT_NCHW) {
                padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
                stride = Stride2D(m_stride4[2], m_stride4[3]);
                dilation = Stride2D(m_dilation4[2], m_dilation4[3]);
                // ksize = Size2D(w.size(2), w.size(3));
            } else if (m_format == FORMAT_NHWC) {
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
                stride = Stride2D(m_stride4[1], m_stride4[2]);
                dilation = Stride2D(m_dilation4[1], m_dilation4[2]);
                // ksize = Size2D(w.size(1), w.size(2));
            }

            {
                stack.push_base(6); // empty base
                need pop_base(&Stack::pop_base, &stack);

                TS_AUTO_CHECK(stack.size() == 0);

                forward(x, w, b, offset, mask, padding, stride, dilation, m_deformable_groups, m_format, out);

                stack.clear();
            }

            return 1;
        }
    }
}
