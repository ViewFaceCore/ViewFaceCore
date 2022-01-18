//
// Created by kier on 2019/9/7.
//

#include "backend/dragon/pooling2d_auto_pad.h"

#include <array>

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "utils/assert.h"

#include "runtime/stack.h"

#include "backend/common_function.h"

#include "core/device.h"
#include "global/operator_factory.h"

namespace ts {
    namespace dragon {

        Pooling2DAutoPad::Pooling2DAutoPad() {
            this->field("ceil", OPTIONAL, tensor::from(true));
            this->field(name::auto_pad, OPTIONAL, tensor::from(name::VALID));
            this->field(name::padding, OPTIONAL, tensor::build(INT32, {4, 2}, {0, 0, 0, 0, 0, 0, 0, 0}));
        }

        void Pooling2DAutoPad::init() {
            supper::init();

            this->ceil_mode = tensor::to_bool(get("ceil"));

            auto auto_pad_str = tensor::to_string(this->get(name::auto_pad));

            if (auto_pad_str == name::VALID) {
                this->auto_pad = AutoPadType::VALID;
            } else if (auto_pad_str == name::NOTSET) {
                this->auto_pad = AutoPadType::VALID;
            } else if (auto_pad_str == name::SAME) {
                this->auto_pad = AutoPadType::SAME_LOWER;
            } else if (auto_pad_str == name::SAME_LOWER) {
                this->auto_pad = AutoPadType::SAME_LOWER;
            } else if (auto_pad_str == name::SAME_UPPER) {
                this->auto_pad = AutoPadType::SAME_UPPER;
            } else {
                TS_LOG_ERROR << "Not supported auto_pad=" << auto_pad_str << eject;
            }

            auto static_padding = tensor::cast(INT32, this->get(name::padding));

            TS_AUTO_CHECK(static_padding.has_shape({4, 2}));

            // IS NCHW format
            this->static_padding.top = static_padding.data<int32_t>()[4];
            this->static_padding.bottom = static_padding.data<int32_t>()[5];
            this->static_padding.left = static_padding.data<int32_t>()[6];
            this->static_padding.right = static_padding.data<int32_t>()[7];
        }

        int Pooling2DAutoPad::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            output.resize(1);
            output[0] = Tensor::Prototype(INT32, {4, 2});
            return 1;
        }

        // VALID: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
        static inline int forward_valid(int x, int ksize, int pad_l, int pad_r, int stride, bool ceil_mode) {
            int y;
            if (ceil_mode) {
                y = int(std::ceil((x + pad_l + pad_r - ksize) / (float) stride) + 1);
            } else {
                y = int(std::floor((x + pad_l + pad_r - ksize) / (float) stride) + 1);
            }
            if ((y - 1) * stride >= (x + pad_l + pad_r)) y--;
            return y;
        }

        static inline Size2D pooling2d_forward_valid(const Size2D &x, const KSize2D &ksize, const Padding2D &padding, const Stride2D &stride, bool ceil_mode) {
            Size2D y;
            y.height = forward_valid(x.height, ksize.height, padding.top, padding.bottom, stride.height, ceil_mode);
            y.width = forward_valid(x.width, ksize.width, padding.left, padding.right, stride.width, ceil_mode);
            return y;
        }

        static inline Padding2D dynamic_padding_valid(const Size2D &input_size, const Padding2D &static_padding, const KSize2D &ksize,
                                                       const Stride2D &stride, bool ceil_mode) {

            Padding2D dynamic_padding;

            Size2D expected_output_size = pooling2d_forward_valid(input_size, ksize, static_padding, stride, ceil_mode);
            Size2D expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride);
            dynamic_padding.top = static_padding.top;
            dynamic_padding.left = static_padding.left;
            dynamic_padding.bottom = static_padding.bottom + expected_input_size.height - input_size.height;
            dynamic_padding.right = static_padding.right + expected_input_size.width - input_size.width;

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

        // SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
        /**
         *
         * @param x
         * @param ksize exclude static padding
         * @param stride
         * @param ceil
         * @return
         */
        static inline _out_pad forward_same(int x, int ksize, int stride, Pooling2DAutoPad::AutoPadType padding, bool ceil_mode) {
            _out_pad out;

            auto idm = x;
            auto odm = int((idm + stride - 1) / float(stride));
            auto padding_needed = std::max<int>(0, (odm - 1) * stride + ksize - idm);

            auto &y = out.out;
            auto &pad_l = out.pad_l;
            auto &pad_r = out.pad_r;

            if (padding == Pooling2DAutoPad::AutoPadType::SAME_UPPER) { DEFINE_SAME_PADDING(pad_l, pad_r); }
            else { DEFINE_SAME_PADDING(pad_r, pad_l); }

            y = int(std::ceil(float(x) / stride));

            return out;
        }

        static inline std::array<_out_pad, 2> pooling2d_forward_same(const Size2D &x, const KSize2D &ksize,
                                                     const Stride2D &stride, Pooling2DAutoPad::AutoPadType padding, bool ceil_mode) {
            std::array<_out_pad, 2> out;
            out[0] = forward_same(x.height, ksize.height, stride.height, padding, ceil_mode);
            out[1] = forward_same(x.width, ksize.width, stride.width, padding, ceil_mode);
            return out;
        }

        static inline Padding2D dynamic_padding_same(const Size2D &input_size, const Padding2D &static_padding, const KSize2D &ksize,
                const Stride2D &stride, Pooling2DAutoPad::AutoPadType padding, bool ceil_mode) {

            Padding2D dynamic_padding;

            auto dynamic_output_size = pooling2d_forward_same(input_size, ksize, stride, padding, ceil_mode);

            Size2D expected_output_size(dynamic_output_size[0].out, dynamic_output_size[1].out);
            Size2D expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride);

            auto padding_height = (expected_input_size.height - input_size.height);
            auto padding_width = (expected_input_size.width - input_size.width);

            dynamic_padding.top = dynamic_output_size[0].pad_l;
            dynamic_padding.left = dynamic_output_size[1].pad_l;
            dynamic_padding.bottom = static_padding.top + static_padding.bottom + (padding_height - dynamic_output_size[0].pad_l);
            dynamic_padding.right = static_padding.left + static_padding.right + (padding_width - dynamic_output_size[1].pad_l);

            return dynamic_padding;
        }

        int Pooling2DAutoPad::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            // get output
            auto &dynamic_padding_tensor = *stack.push(INT32, {4, 2}, MemoryDevice(CPU));

            // convert input
            auto shape = stack[0].sizes();
            auto ksize_tensor = tensor::cast(INT32, stack[1]);
            auto stride_tensor = tensor::cast(INT32, stack[2]);

            Padding2D dynamic_padding;

            KSize2D ksize;
            Stride2D stride;
            Size2D input_size;

            ksize.height = ksize_tensor.data<int32_t>()[2];
            ksize.width = ksize_tensor.data<int32_t>()[3];
            stride.height = stride_tensor.data<int32_t>()[2];
            stride.width = stride_tensor.data<int32_t>()[3];
            input_size.height = shape[2];
            input_size.width = shape[3];

            switch (this->auto_pad) {
                case AutoPadType::NOTSET:
                    dynamic_padding = dynamic_padding_valid(input_size, static_padding, ksize, stride, ceil_mode);
                    break;
                case AutoPadType::VALID:
                    dynamic_padding = dynamic_padding_valid(input_size, static_padding, ksize, stride, ceil_mode);
                    break;
                case AutoPadType::SAME_UPPER:
                    dynamic_padding = dynamic_padding_same(input_size, static_padding, ksize, stride,
                            AutoPadType::SAME_UPPER, ceil_mode);
                    break;
                case AutoPadType::SAME_LOWER:
                    dynamic_padding = dynamic_padding_same(input_size, static_padding, ksize, stride,
                            AutoPadType::SAME_LOWER, ceil_mode);
                    break;
            }

            auto dynamic_padding_data = dynamic_padding_tensor.data<int32_t>();

            dynamic_padding_data[0] = 0;
            dynamic_padding_data[1] = 0;
            dynamic_padding_data[2] = 0;
            dynamic_padding_data[3] = 0;
            dynamic_padding_data[4] = dynamic_padding.top;
            dynamic_padding_data[5] = dynamic_padding.bottom;
            dynamic_padding_data[6] = dynamic_padding.left;
            dynamic_padding_data[7] = dynamic_padding.right;

            return 1;
        }
    }
}

using namespace ts;
using namespace ts::dragon;

TS_REGISTER_OPERATOR(Pooling2DAutoPad, CPU, "_dragon_pooling2d_padding")
