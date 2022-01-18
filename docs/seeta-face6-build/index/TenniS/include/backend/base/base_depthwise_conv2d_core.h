//
// Created by kier on 2019/2/16.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_CORE_H
#define TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_CORE_H

#include "base_conv2d_core.h"

namespace ts {
    namespace base {
        using DepthwiseConv2DCore = Conv2DCore;

        template <typename Conv2D, typename Core>
        using DepthwiseConv2DWithCore = Conv2DWithCore<Conv2D, Core>;

        template <typename Conv2D, typename Core>
        using PackedDepthwiseConv2DWithCore = PackedConv2DWithCore<Conv2D, Core>;
    }
}

#endif //TENSORSTACK_BACKEND_BASE_BASE_DEPTHWISE_CONV2D_CORE_H
