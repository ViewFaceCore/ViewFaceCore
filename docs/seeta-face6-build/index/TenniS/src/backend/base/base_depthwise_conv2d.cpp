//
// Created by kier on 2019/2/16.
//

#include <backend/base/base_depthwise_conv2d.h>

#include "backend/base/base_depthwise_conv2d.h"

namespace ts {
    namespace base {
        int DepthwiseConv2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            supper::infer(stack, output);

            auto &x = stack[0];
            auto &w = stack[1];

            TS_AUTO_CHECK(w.size(0) == 1);

            auto output_shape = output[0].sizes();
            output_shape[1] = x.size(1) * w.size(0);

            output[0] = Tensor::Prototype(output[0].dtype(), output_shape);

            return 1;
        }
    }
}
