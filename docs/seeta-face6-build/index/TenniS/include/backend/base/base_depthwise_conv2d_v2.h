//
// Created by kier on 2019/2/16.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_V2_H
#define TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_V2_H

#include "base_conv2d_v2.h"

namespace ts {
    namespace base {
        class DepthwiseConv2DV2 : public Conv2DV2 {
        public:
            using self = DepthwiseConv2DV2;
            using supper = Conv2DV2;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_V2_H
