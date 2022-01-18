#ifndef TENSORSTACK_KERNELS_GPU_POOLING2D_H
#define TENSORSTACK_KERNELS_GPU_POOLING2D_H

#include "operator_on_gpu.h"
#include "backend/base/base_pooling2d.h"
#include "pooling2d_core.h"


namespace ts {
    namespace gpu {
        using Pooling2D = base::Pooling2DWithCore<OperatorOnGPU<base::Pooling2D>, Pooling2DCore>;
    }
}


#endif //TENSORSTACK_KERNELS_GPU_POOLING2D_H


