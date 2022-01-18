//
// Created by kier on 2019/2/16.
//

#include <backend/base/base_depthwise_conv2d_v2.h>

#include "backend/base/base_depthwise_conv2d_v2.h"

namespace ts {
    namespace base {
        int DepthwiseConv2DV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            supper::infer(stack, output);

            auto &x = stack[0];
            auto &w = stack[2];

            TS_AUTO_CHECK(w.size(0) == 1);

            auto output_shape = output[0].sizes();
            output_shape[1] = x.size(1) * w.size(0);

            output[0] = Tensor::Prototype(output[0].dtype(), output_shape);

            return 1;
        }
    }
}
