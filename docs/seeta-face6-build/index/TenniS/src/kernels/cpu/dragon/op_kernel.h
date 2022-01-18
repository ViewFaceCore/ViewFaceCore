//
// Created by kier on 2019/9/7.
//

#ifndef DRAGON_UTILS_OP_KERNEL_H
#define DRAGON_UTILS_OP_KERNEL_H

namespace ts {
    namespace dragon {
        namespace kernel {
/*! vision.roi_align */
            template<typename T, class Context>
            void ROIAlign(
                    const int C,
                    const int H,
                    const int W,
                    const int pool_h,
                    const int pool_w,
                    const int num_rois,
                    const float spatial_scale,
                    const int sampling_ratio,
                    const T *x,
                    const float *rois,
                    T *y,
                    Context *ctx);
        }
    }
}


#endif //DRAGON_UTILS_OP_KERNEL_H
