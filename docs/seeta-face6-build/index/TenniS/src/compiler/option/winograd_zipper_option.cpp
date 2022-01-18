//
// Created by kier on 2019-06-10.
//

#include "compiler/option/winograd_zipper_option.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "module/menu.h"

#include <valarray>
#include "kernels/common/function.h"
#include "backend/common_function.h"

namespace ts {
    bool Conv2dZipperOption::zip(const ComputingDevice &device, Node node, Node &zipped_node) const {
        if (device.type() != CPU)
            return false;

        auto bubble = node.bubble();
        auto op_name = bubble.op();
        if (op_name != name::layer::conv2d() && op_name != name::layer::conv2d_v2())
            return false;

        auto format_tensor = bubble.get(name::format);
        auto format = tensor::to_string(format_tensor);
        if(format != name::NCHW)
            return false;
        auto stride_tensor = tensor::cast(INT32, bubble.get(name::stride));

        std::valarray<int> dilation4;
        dilation4.resize(4);
        std::valarray<int> stride4;
        stride4.resize(4);

        if (bubble.has(name::dilation)) {
            auto dilation_tensor = tensor::cast(INT32, bubble.get(name::dilation));
            for (size_t i = 0; i < 4; ++i)
                dilation4[i] = dilation_tensor.data<int32_t>(i);
        }

        for (size_t i = 0; i < 4; ++i)
            stride4[i] = stride_tensor.data<int32_t>(i);

        auto inputs = node.inputs();

        KSize2D stride_size; Dilation2D dilation_size;
        stride_size.height = stride4[2];
        stride_size.width = stride4[3];
        dilation_size.height = dilation4[2];
        dilation_size.width = dilation4[3];

        Tensor kernel_tensor;
        Tensor padding_tensor;
        Tensor padding_val_tensor;
        if (bubble.has(name::padding_value)) {
            padding_val_tensor = bubble.get(name::padding_value);
        }
        if (op_name == name::layer::conv2d()) {
            TS_AUTO_CHECK(inputs.size() == 2);
            kernel_tensor = inputs[1].bubble().get(name::value);
            padding_tensor = bubble.get(name::padding);
        }
        else {
            TS_AUTO_CHECK(inputs.size() == 3);
            kernel_tensor = inputs[2].bubble().get(name::value);
        }
        auto kernel_shape = kernel_tensor.sizes();
        // Shape padding_shape = padding_tensor.sizes();

        bool winograd_flag = KernelCommonFunc<float>::winograd_check(kernel_shape,
                                                                     stride_size,
                                                                     dilation_size);

        if(!winograd_flag)
            return false;

        if (op_name == name::layer::conv2d()) {
            //std::string winograd_name = node.bubble().name() + "_conv2d_winograd";
            std::string winograd_name = node.bubble().name();
            zipped_node = bubble::op(winograd_name, name::layer::conv2d_winograd(), { inputs[0], inputs[1] });
            zipped_node.bubble().set(name::padding, padding_tensor);
        }
        else {
            //std::string winograd_name = node.bubble().name() + "_conv2d_winograd";
            std::string winograd_name = node.bubble().name();
            zipped_node = bubble::op(winograd_name, name::layer::conv2d_winograd_v2(), { inputs[0], inputs[1], inputs[2] });
        }

        zipped_node.bubble().set(name::format, format_tensor);
        if (bubble.has(name::padding_value)) {
            zipped_node.bubble().set(name::padding_value, padding_val_tensor);
        }

        return true;
    }
}

//#ifdef TS_USE_WINOGRAD
//TS_REGISTER_ZIPPER_OPTION(ts::Conv2dZipperOption)
//#endif
