#ifndef TENSORSTACK_KERNELS_CPU_PAD2D_ALGORITHM_H
#define TENSORSTACK_KERNELS_CPU_PAD2D_ALGORITHM_H

#include <core/tensor.h>

namespace ts{
    namespace cpu{

        template<typename T>
        class TS_DEBUG_API PadAlgorithm {
        public:

            //NOTE:Only supports pad on NCHW or NHWC formats.
            static void pad_nchw_nhwc(const Tensor &x,
                                 const std::vector<std::array<int, 2>> &padding,
                                 float padding_value,
                                 Tensor &out);

            //NOTE:pad2d_Superseded and cut2d_Superseded only support NCHW pad on [H,W] now.
            //pad2d don't support negative pad,cut2d support negative pad only.
            static void pad2d(const Tensor &x,
                                         const std::array<int, 2> &padding_h,
                                         const std::array<int, 2> &padding_w,
                                         float padding_value,
                                         Tensor &out);

            static void cut2d(const Tensor &x,
                                         const std::array<int, 2> &padding_h,
                                         const std::array<int, 2> &padding_w,
                                         float padding_value,
                                         Tensor &out);
        };
    }//cpu
}//ts

#endif