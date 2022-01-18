//
// Created by kier on 2019/2/19.
//

#ifndef TENSORSTACK_KERNELS_GPU_POOLING2D_CORE_H
#define TENSORSTACK_KERNELS_GPU_POOLING2D_CORE_H

#include "backend/base/base_pooling2d_core.h"

namespace ts {
    namespace gpu {
        class Pooling2DCore : public base::Pooling2DCore {
        public:
            void pooling2d(const Tensor &x, Pooling2DType type,
                           const Padding2D &padding, Padding2DType padding_type,
                           const Size2D &ksize, const Stride2D &stride,
                           Conv2DFormat format, Tensor &out) override;
        };
    }
}


#endif //TENSORSTACK_KERNELS_GPU_POOLING2D_CORE_H
