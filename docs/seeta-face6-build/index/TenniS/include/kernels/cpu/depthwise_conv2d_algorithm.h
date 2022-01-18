#ifndef TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_ALGORITHM_H
#define TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_ALGORITHM_H

#include "core/tensor.h"
#include <backend/common_structure.h>

namespace ts{
    namespace cpu{
        template<typename T>
        class TS_DEBUG_API DepthwiseConv2dAlgorithm {
        public:

            static void depthwise_general(
                const Tensor &x,
                const Padding2D &padding,
                float padding_value,
                const Tensor &weight,
                const Stride2D &stride,
                const Dilation2D &dilation,
                Tensor &out);

            static void depthwise_3x3_s1(
                const Tensor &x, 
                const Padding2D &padding, 
                float padding_value,
                const Tensor &weight, 
                const Stride2D &stride,
                const Dilation2D &dilation,
                Tensor &out);

            static void depthwise_3x3_s2(
                const Tensor &x,
                const Padding2D &padding,
                float padding_value,
                const Tensor &weight,
                const Stride2D &stride,
                const Dilation2D &dilation,
                Tensor &out);

        };
    }
}

#endif //TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_ALGORITHM_H