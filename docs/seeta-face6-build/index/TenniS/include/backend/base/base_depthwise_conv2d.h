//
// Created by kier on 2019/2/16.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_H
#define TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_H

#include "base_conv2d.h"

namespace ts {
    namespace base {
        class DepthwiseConv2D : public Conv2D {
        public:
            using self = DepthwiseConv2D;
            using supper = Conv2D;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_H
