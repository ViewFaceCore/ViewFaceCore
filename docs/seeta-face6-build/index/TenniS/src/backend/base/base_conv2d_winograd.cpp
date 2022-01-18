#include <backend/base/base_conv2d_winograd.h>

#include "backend/base/base_conv2d_winograd.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

#include "backend/common_function.h"
#include "utils/need.h"

#include "kernels/common/function.h"

namespace ts {
    namespace base{
        Conv2DWinograd::Conv2DWinograd() {
            field(name::format, REQUIRED);
            field(name::padding, REQUIRED);
            field(name::padding_value, OPTIONAL, tensor::from(0.0f));
            field(name::kernel_winograd_transformed, OPTIONAL, tensor::from<bool>(false));
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

        void Conv2DWinograd::init() {
            supper::init();

            auto format = tensor::to_string(get(name::format));
            auto padding_tensor = tensor::cast(INT32, get(name::padding));
            m_padding_value = tensor::to_float(get(name::padding_value));

            if (has(name::kernel_winograd_transformed)) {
                m_kernel_transformed = tensor::to_bool(get(name::kernel_winograd_transformed));
            }

            TS_AUTO_CHECK(padding_tensor.has_shape({ 4, 2 }));

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

            // only support native conv2d
            if (m_format == FORMAT_NCHW) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[2] != 0 ||
                    m_padding4x2[3] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
            }
            else if (m_format == FORMAT_NHWC) {
                if (m_padding4x2[0] != 0 ||
                    m_padding4x2[1] != 0 ||
                    m_padding4x2[6] != 0 ||
                    m_padding4x2[7] != 0) {
                    TS_LOG_ERROR << this->op() << " do not support padding: " << to_string(m_padding4x2) << eject;
                }
            }

        }

        int Conv2DWinograd::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x_tensor = stack[0];
            auto k_tensor = stack[1];

            TS_AUTO_CHECK(x_tensor.dims() == 4);
            TS_AUTO_CHECK(k_tensor.dims() == 4);

            Size2D x;
            //winograd conv2d only suppot ksize == 3x3 && stride == 1 && dialations == 1
            Size2D ksize(3, 3);
            Stride2D stride(1, 1);
            Dilation2D dilations(1, 1);
//            Padding2D padding(0, 0, 0, 0);
            Padding2D padding;
            if (m_format == FORMAT_NCHW) {
                x = Size2D(x_tensor.size(2), x_tensor.size(3));
                padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            }
            else if (m_format == FORMAT_NHWC) {
                x = Size2D(x_tensor.size(1), x_tensor.size(2));
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
            }

            Size2D y = conv2d_forward(x, padding, ksize, stride, dilations);

            Tensor::Prototype out_proto;

            if (m_format == FORMAT_NCHW) {
                out_proto = Tensor::Prototype(
                    x_tensor.dtype(),
                    { x_tensor.size(0), k_tensor.size(0), y.height, y.width });
            }
            else if (m_format == FORMAT_NHWC) {
                out_proto = Tensor::Prototype(
                    x_tensor.dtype(),
                    { x_tensor.size(0), y.height, y.width, k_tensor.size(0) });
            }

            output.resize(1);
            output[0] = out_proto;

            return 1;
        }

        int Conv2DWinograd::run(ts::Stack &stack) {
            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto memory_device = running_memory_device();

            auto x_tensor = stack[0].view(memory_device);
            auto kernel_tensor = stack[1].view(memory_device);

            auto out = *stack.push(output[0], memory_device);

            Padding2D padding;

            if (m_format == FORMAT_NCHW) {
                padding = Padding2D(m_padding4x2[4], m_padding4x2[5], m_padding4x2[6], m_padding4x2[7]);
            }
            else if (m_format == FORMAT_NHWC) {
                padding = Padding2D(m_padding4x2[2], m_padding4x2[3], m_padding4x2[4], m_padding4x2[5]);
            }

            //warm up
            if(!m_kernel_transformed || m_k_transformed.empty()){
                //select winograd mode
                WinogradConv2DMode winograd_mode;
                KernelCommonFunc<float>::winograd_mode_select_on_arm(x_tensor.sizes(), kernel_tensor.size(0), winograd_mode);
                m_winograd_mode = winograd_mode;

                Shape kernel_shape = kernel_tensor.sizes();
                Shape kernel_transformed_shape;
                int in_tile_width = winograd_mode == F2X2_3X3 ? 4 : 8;
                int in_tile_height = in_tile_width;
                kernel_transformed_shape = {kernel_shape[0], kernel_shape[1], in_tile_height, in_tile_width };
                m_k_transformed = Tensor(Tensor::InFlow::HOST, kernel_tensor.dtype(), kernel_transformed_shape);

                conv2d_tranform_kernel(m_winograd_mode, kernel_tensor, m_k_transformed);

                m_kernel_transformed = true;
            }

            conv2d_winograd(x_tensor, m_winograd_mode, padding, m_padding_value, m_k_transformed, m_format, out, m_kernel_transformed);


            return 1;
        }
    }
}