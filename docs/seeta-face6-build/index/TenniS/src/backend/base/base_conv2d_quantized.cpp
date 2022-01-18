#include <backend/base/base_conv2d_quantized.h>

#include "backend/base/base_conv2d_quantized.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

#include "backend/common_function.h"
#include "utils/need.h"

namespace ts {
    namespace base {
        Conv2DQuantized::Conv2DQuantized() {
            field(name::format, REQUIRED);
            field(name::padding, REQUIRED);
            field(name::padding_value, OPTIONAL, tensor::from(0.0f));
            field(name::stride, REQUIRED);
            field(name::dilation, OPTIONAL);
            field(name::typo::dialations, OPTIONAL);

            //field(name::quantize_scale, REQUIRED);
            field(name::dequantize_scales, REQUIRED);
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

        void Conv2DQuantized::init() {
            supper::init();

            auto format = tensor::to_string(get(name::format));
            auto padding_tensor = tensor::cast(INT32, get(name::padding));
            m_padding_value = tensor::to_float(get(name::padding_value));
            auto stride_tensor = tensor::cast(INT32, get(name::stride));

            Tensor dilation_tensor;
            if (has(name::dilation)) {
                dilation_tensor = tensor::cast(INT32, get(name::dilation));
            }
            else if (has(name::typo::dialations)) {
                dilation_tensor = tensor::cast(INT32, get(name::typo::dialations));
            }

            if (dilation_tensor.empty()) {
                TS_LOG_ERROR << this->op() << " must set " << name::dilation << " or " << name::typo::dialations << eject;
            }

            TS_AUTO_CHECK(padding_tensor.has_shape({ 4, 2 }));
            TS_AUTO_CHECK(stride_tensor.has_shape({ 4, }));
            TS_AUTO_CHECK(dilation_tensor.has_shape({ 4, }));

            if (format == name::NCHW) {
                m_format = FORMAT_NCHW;
            }
            else if (format == name::NHWC) {
                m_format = FORMAT_NHWC;
            }
            else {
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
            }
            else if (m_format == FORMAT_NHWC) {
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

            //m_quantize_scale = tensor::to_float(get(name::quantize_scale));
            auto dequantize_scale_tensor = get(name::dequantize_scales);
            m_dequantize_scales.resize(dequantize_scale_tensor.count());
            for (int i = 0; i < dequantize_scale_tensor.count(); i++){
                m_dequantize_scales[i] = tensor::cast(FLOAT32, dequantize_scale_tensor).data<float>()[i];
            }

        }

        int Conv2DQuantized::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x_tensor = stack[0];
            auto w_tensor = stack[1];

            TS_AUTO_CHECK(x_tensor.dims() == 4);
            TS_AUTO_CHECK(w_tensor.dims() == 4);

            TS_AUTO_CHECK(x_tensor.dtype() == w_tensor.dtype());

            TS_AUTO_CHECK(w_tensor.size(1) == x_tensor.size(1));

            // input type should be int8
            TS_AUTO_CHECK(x_tensor.dtype() == INT8);
            // dequantize_scale num should be equale to kernel num;
            TS_AUTO_CHECK(m_dequantize_scales.size() == w_tensor.size(0));

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
            }
            else if (m_format == FORMAT_NHWC) {
                x = Size2D(x_tensor.size(1), x_tensor.size(2));
                ksize = Size2D(w_tensor.size(1), w_tensor.size(2));
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
                stride = Stride2D(m_stride4[1], m_stride4[2]);
                dialations = Stride2D(m_dilation4[1], m_dilation4[2]);
            }

            Size2D y = conv2d_forward(x, padding, ksize, stride, dialations);

            Tensor::Prototype out_proto;

            //TO DO:support requantize,so output type can be int8 too.
            if (m_format == FORMAT_NCHW) {
                out_proto = Tensor::Prototype(
                    FLOAT32,
                    { x_tensor.size(0), w_tensor.size(0), y.height, y.width });
            }
            else if (m_format == FORMAT_NHWC) {
                out_proto = Tensor::Prototype(
                    FLOAT32,
                    { x_tensor.size(0), y.height, y.width, w_tensor.size(0) });
            }

            output.resize(1);
            output[0] = out_proto;

            return 1;
        }

        int Conv2DQuantized::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            Tensor x = stack[0].view(memory_device);
            Tensor w = stack[1].view(memory_device);

            Tensor out = *stack.push(output_protos[0], memory_device);

            Padding2D padding;
            Stride2D stride;
            Dilation2D dilation;

            if (m_format == FORMAT_NCHW) {
                padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
                stride = Stride2D(m_stride4[2], m_stride4[3]);
                dilation = Stride2D(m_dilation4[2], m_dilation4[3]);
            }
            else if (m_format == FORMAT_NHWC) {
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
                stride = Stride2D(m_stride4[1], m_stride4[2]);
                dilation = Stride2D(m_dilation4[1], m_dilation4[2]);
            }

            {
                stack.push_base(3); // empty base
                need pop_base(&Stack::pop_base, &stack);

                TS_AUTO_CHECK(stack.size() == 0);

                conv2d(x, padding, m_padding_value, w, stride, dilation, m_format, m_dequantize_scales, out, stack);

                stack.clear();
            }

            return 1;
        }
    }
}

