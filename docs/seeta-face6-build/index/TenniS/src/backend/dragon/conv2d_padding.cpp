//
// Created by kier on 2019/9/9.
//

#include "backend/dragon/conv2d_padding.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "runtime/stack.h"
#include "backend/common_function.h"
#include <array>

namespace ts {
    namespace dragon {
        Conv2DPadding::Conv2DPadding() {
            field(name::format, REQUIRED);
            field(name::stride, REQUIRED);
            field(name::dilation, OPTIONAL);
            this->field(name::padding_method, OPTIONAL, tensor::from(name::VALID));
            this->field(name::padding, OPTIONAL, tensor::build(INT32, {4, 2}, {0, 0, 0, 0, 0, 0, 0, 0}));
        }


        void Conv2DPadding::init() {
            supper::init();

            auto padding_method_str = tensor::to_string(this->get(name::padding_method));

            if (padding_method_str == name::SAME) {
                this->m_padding_method = PaddingMethod::SAME_LOWER;
            } else if (padding_method_str == name::VALID) {
                this->m_padding_method = PaddingMethod::VALID;
            } else if (padding_method_str == name::NOTSET) {
                this->m_padding_method = PaddingMethod::VALID;
            } else if (padding_method_str == name::SAME_UPPER) {
                this->m_padding_method = PaddingMethod::SAME_UPPER;
            } else if (padding_method_str == name::SAME_LOWER) {
                this->m_padding_method = PaddingMethod::SAME_LOWER;
            } else {
                TS_LOG_ERROR << "Not supported padding_method=" << padding_method_str << eject;
            }

            auto format_str = tensor::to_string(get(name::format));

            if (format_str == name::NCHW) {
                this->m_format = Conv2DFormat::FORMAT_NCHW;
            } else if (format_str == name::NHWC) {
                this->m_format = Conv2DFormat::FORMAT_NHWC;
            } else {
                TS_LOG_ERROR << "Not supported format=" << format_str << eject;
            }

            auto static_padding = tensor::cast(INT32, this->get(name::padding));
            TS_AUTO_CHECK(static_padding.has_shape(4, 2));
            auto dilation = tensor::cast(INT32, get(name::dilation));
            TS_AUTO_CHECK(dilation.has_shape(4));
            auto stride = tensor::cast(INT32, get(name::stride));
            TS_AUTO_CHECK(stride.has_shape(4));

            if (m_format == Conv2DFormat::FORMAT_NCHW) {
                TS_AUTO_CHECK(static_padding.data<int32_t>()[0] == 0
                              || static_padding.data<int32_t>()[1] == 0
                              || static_padding.data<int32_t>()[2] == 0
                              || static_padding.data<int32_t>()[3] == 0);
                this->m_static_padding.top = static_padding.data<int32_t>()[4];
                this->m_static_padding.bottom = static_padding.data<int32_t>()[5];
                this->m_static_padding.left = static_padding.data<int32_t>()[6];
                this->m_static_padding.right = static_padding.data<int32_t>()[7];
                TS_AUTO_CHECK(dilation.data<int32_t>()[0] == 1
                              || dilation.data<int32_t>()[1] == 1);
                this->m_dilation.height = dilation.data<int32_t>()[2];
                this->m_dilation.width = dilation.data<int32_t>()[3];
                TS_AUTO_CHECK(stride.data<int32_t>()[0] == 1
                              || stride.data<int32_t>()[1] == 1);
                this->m_stride.height = stride.data<int32_t>()[2];
                this->m_stride.width = stride.data<int32_t>()[3];
            } else if (m_format == Conv2DFormat::FORMAT_NHWC) {
                TS_AUTO_CHECK(static_padding.data<int32_t>()[0] == 0
                              || static_padding.data<int32_t>()[1] == 0
                              || static_padding.data<int32_t>()[6] == 0
                              || static_padding.data<int32_t>()[7] == 0);
                this->m_static_padding.top = static_padding.data<int32_t>()[2];
                this->m_static_padding.bottom = static_padding.data<int32_t>()[3];
                this->m_static_padding.left = static_padding.data<int32_t>()[4];
                this->m_static_padding.right = static_padding.data<int32_t>()[5];
                TS_AUTO_CHECK(dilation.data<int32_t>()[0] == 1
                              || dilation.data<int32_t>()[3] == 1);
                this->m_dilation.height = dilation.data<int32_t>()[1];
                this->m_dilation.width = dilation.data<int32_t>()[2];
                TS_AUTO_CHECK(stride.data<int32_t>()[0] == 1
                              || stride.data<int32_t>()[3] == 1);
                this->m_stride.height = stride.data<int32_t>()[1];
                this->m_stride.width = stride.data<int32_t>()[2];
            }
        }

        int Conv2DPadding::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);
            output.resize(1);
            output[0] = Tensor::Prototype(INT32, {4, 2});
            return 1;
        }

        // pretty same as TS
        static inline Padding2D
        dynamic_padding_valid(const Size2D &input_size, const Padding2D &static_padding, const KSize2D &ksize,
                              const Stride2D &stride, const Dilation2D &dilation) {


            Padding2D dynamic_padding;

            dynamic_padding.top = static_padding.top;
            dynamic_padding.left = static_padding.left;
            dynamic_padding.bottom = static_padding.bottom;
            dynamic_padding.right = static_padding.right;

            return dynamic_padding;
        }

        struct _out_pad {
            _out_pad() = default;

            int out;    // output size
            int pad_l;    // final padding (including static padding and dynamic padding)
            int pad_r;    // final padding (including static padding and dynamic padding)
        };


#define DEFINE_SAME_PADDING(A, B) \
    A = padding_needed / 2; \
    B = padding_needed - A

        static inline _out_pad
        forward_same(int x, int ksize, int stride, int dilation, Conv2DPadding::PaddingMethod padding) {
            _out_pad out;

            auto idm = x;
            const int dk = dilation * (ksize - 1) + 1;
            int odm = int((idm + stride - 1) / float(stride));
            int padding_needed = std::max<int>(0, (odm - 1) * stride + dk - idm);

            auto &y = out.out;
            auto &pad_l = out.pad_l;
            auto &pad_r = out.pad_r;

            if (padding != Conv2DPadding::PaddingMethod::SAME_UPPER) { DEFINE_SAME_PADDING(pad_l, pad_r); }
            else { DEFINE_SAME_PADDING(pad_r, pad_l); }  // SAME_LOWER or SAME

            y = odm;

            return out;
        }

        static inline std::array<_out_pad, 2>
        conv2d_forward_same(const Size2D &x, const Padding2D &padding, const KSize2D &ksize,
                            const Stride2D &stride, const Dilation2D &dilation,
                            Conv2DPadding::PaddingMethod padding_method) {
            std::array<_out_pad, 2> out;
            out[0] = forward_same(x.height, ksize.height, stride.height, dilation.height, padding_method);
            out[1] = forward_same(x.width, ksize.width, stride.width, dilation.width, padding_method);
            return out;
        }

        static inline Padding2D
        dynamic_padding_same(const Size2D &input_size, const Padding2D &static_padding, const KSize2D &ksize,
                             const Stride2D &stride, const Dilation2D &dilation,
                             Conv2DPadding::PaddingMethod padding_method) {

            Padding2D dynamic_padding;

            auto dynamic_output_size = conv2d_forward_same(input_size, static_padding, ksize, stride, dilation,
                                                           padding_method);

            Size2D expected_output_size(dynamic_output_size[0].out, dynamic_output_size[1].out);
            Size2D expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride);

            auto padding_height = (expected_input_size.height - input_size.height);
            auto padding_width = (expected_input_size.width - input_size.width);

            dynamic_padding.top = dynamic_output_size[0].pad_l;
            dynamic_padding.left = dynamic_output_size[1].pad_l;
            dynamic_padding.bottom =
                    static_padding.top + static_padding.bottom + (padding_height - dynamic_output_size[0].pad_l);
            dynamic_padding.right =
                    static_padding.left + static_padding.right + (padding_width - dynamic_output_size[1].pad_l);

            return dynamic_padding;
        }

        int Conv2DPadding::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x_tensor = stack[0];
            auto &w_tensor = stack[1];

            TS_AUTO_CHECK(x_tensor.dims() == 4);
            TS_AUTO_CHECK(w_tensor.dims() == 4);

            TS_AUTO_CHECK(x_tensor.dtype() == w_tensor.dtype());

            TS_AUTO_CHECK(w_tensor.size(1) == x_tensor.size(1));

            Size2D x;
            Size2D ksize;

            if (m_format == FORMAT_NCHW) {
                x = Size2D(x_tensor.size(2), x_tensor.size(3));
                ksize = Size2D(w_tensor.size(2), w_tensor.size(3));
            } else if (m_format == FORMAT_NHWC) {
                x = Size2D(x_tensor.size(1), x_tensor.size(2));
                ksize = Size2D(w_tensor.size(1), w_tensor.size(2));
            }

            Padding2D dynamic_padding;

            if (m_padding_method == PaddingMethod::SAME_UPPER) {
                dynamic_padding = dynamic_padding_same(x, m_static_padding, ksize, m_stride, m_dilation,
                                                       m_padding_method);
            } else if (m_padding_method == PaddingMethod::SAME_LOWER) {
                dynamic_padding = dynamic_padding_same(x, m_static_padding, ksize, m_stride, m_dilation,
                                                       m_padding_method);
            } else if (m_padding_method == PaddingMethod::VALID) {
                dynamic_padding = dynamic_padding_valid(x, m_static_padding, ksize, m_stride, m_dilation);
            } else {
                TS_LOG_ERROR << "What a Terrible Failure!" << eject;
            }

            // get output
            auto &dynamic_padding_tensor = *stack.push(INT32, {4, 2}, MemoryDevice(CPU));
            auto dynamic_padding_data = dynamic_padding_tensor.data<int32_t>();

            if (m_format == FORMAT_NCHW) {
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
                dynamic_padding_data[2] = dynamic_padding.top;
                dynamic_padding_data[3] = dynamic_padding.bottom;
                dynamic_padding_data[4] = dynamic_padding.left;
                dynamic_padding_data[5] = dynamic_padding.right;
                dynamic_padding_data[6] = 0;
                dynamic_padding_data[7] = 0;
            }

            return 1;
        }
    }
}

using namespace ts;
using namespace ts::dragon;

TS_REGISTER_OPERATOR(Conv2DPadding, CPU, "_dragon_conv2d_padding")
