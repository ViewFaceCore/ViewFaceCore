//
// Created by kier on 2019/3/17.
//

#include "backend/tf/conv2d_padding.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "runtime/stack.h"
#include "backend/common_function.h"

namespace ts {
    namespace tf {
        Conv2DPadding::Conv2DPadding() {
            field(name::format, REQUIRED);
            field(name::stride, REQUIRED);
            field(name::dilation, OPTIONAL);
            this->field(name::padding_method, REQUIRED);
            this->field(name::padding, OPTIONAL, tensor::build(INT32, {4, 2}, {0, 0, 0, 0, 0, 0, 0, 0}));
        }


        void Conv2DPadding::init() {
            supper::init();

            auto padding_method_str = tensor::to_string(this->get(name::padding_method));

            if (padding_method_str == name::SAME) {
                this->m_padding_method = PaddingMethod::SAME;
            } else if (padding_method_str == name::VALID) {
                this->m_padding_method = PaddingMethod::VALID;
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

        // VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
        static inline int forward_valid(int input_spatial_shape, int kernel_spatial_shape, int strides_spatial_shape) {
            return int(std::ceil((input_spatial_shape - kernel_spatial_shape + 1) / float(strides_spatial_shape)));
        }

        static inline Size2D conv2d_forward_valid(const Size2D &x, const Padding2D &padding, const KSize2D &ksize,
                                                  const Stride2D &stride, const Dilation2D &dilation) {
            auto this_kernel_height = (ksize.height - 1) * dilation.height + 1;
            auto this_kernel_width = (ksize.width - 1) * dilation.width + 1;
            Size2D y;
            y.height = forward_valid(x.height + padding.top + padding.bottom, this_kernel_height, stride.height);
            y.width = forward_valid(x.width + padding.left + padding.right, this_kernel_width, stride.width);
            return y;
        }

        static inline Padding2D dynamic_padding_valid(const Size2D &input_size, const Padding2D &static_padding, const KSize2D &ksize,
                                                     const Stride2D &stride, const Dilation2D &dilation) {


            Padding2D dynamic_padding;

            Size2D expected_output_size = conv2d_forward_valid(input_size, static_padding, ksize, stride, dilation);
            Size2D expected_input_size = conv2d_backward(expected_output_size, static_padding, ksize, stride, dilation);
            dynamic_padding.top = static_padding.top;
            dynamic_padding.left = static_padding.left;
            dynamic_padding.bottom = static_padding.bottom + expected_input_size.height - input_size.height;
            dynamic_padding.right = static_padding.right + expected_input_size.width - input_size.width;

            return dynamic_padding;
        }

        // SAME: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
        static inline int forward_same(int input_spatial_shape, int kernel_spatial_shape, int strides_spatial_shape) {
            return int(std::ceil(input_spatial_shape / float(strides_spatial_shape)));
        }

        static inline Size2D conv2d_forward_same(const Size2D &x, const Padding2D &padding, const KSize2D &ksize,
                                                 const Stride2D &stride, const Dilation2D &dilation) {
            auto this_kernel_height = (ksize.height - 1) * dilation.height + 1;
            auto this_kernel_width = (ksize.width - 1) * dilation.width + 1;
            Size2D y;
            y.height = forward_same(x.height + padding.top + padding.bottom, this_kernel_height, stride.height);
            y.width = forward_same(x.width + padding.left + padding.right, this_kernel_width, stride.width);
            return y;
        }

        static inline Padding2D dynamic_padding_same(const Size2D &input_size, const Padding2D &static_padding, const KSize2D &ksize,
                                                     const Stride2D &stride, const Dilation2D &dilation) {

            Padding2D dynamic_padding;

            Size2D expected_output_size = conv2d_forward_same(input_size, static_padding, ksize, stride, dilation);
            Size2D expected_input_size = conv2d_backward(expected_output_size, static_padding, ksize, stride, dilation);

            auto padding_height = (expected_input_size.height - input_size.height);
            auto padding_width = (expected_input_size.width - input_size.width);
            auto half_padding_height = padding_height / 2;
            auto half_padding_width = padding_width / 2;

            dynamic_padding.top = static_padding.top + half_padding_height;
            dynamic_padding.left = static_padding.left + half_padding_width;
            dynamic_padding.bottom = static_padding.bottom + (padding_height - half_padding_height);
            dynamic_padding.right = static_padding.right + (padding_width - half_padding_width);

            return dynamic_padding;
        }

        int Conv2DPadding::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x_tensor = stack[0];
            auto &w_tensor = stack[1];

            TS_AUTO_CHECK(x_tensor.dims() == 4);
            TS_AUTO_CHECK(w_tensor.dims() == 4);

            TS_AUTO_CHECK(x_tensor.dtype() == w_tensor.dtype());

            if (w_tensor.size(1) != x_tensor.size(1)) {
                TS_LOG_ERROR << "Conv2D failed: x=" << x_tensor.proto() << ", w=" << w_tensor.proto()
                    << ", format=" << tensor::to_string(get(name::format)) << "." << eject;
            }

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

            if (m_padding_method == PaddingMethod::SAME) {
                dynamic_padding = dynamic_padding_same(x, m_static_padding, ksize, m_stride, m_dilation);
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
using namespace ts::tf;

TS_REGISTER_OPERATOR(Conv2DPadding, CPU, name::layer::tf_conv2d_padding())
