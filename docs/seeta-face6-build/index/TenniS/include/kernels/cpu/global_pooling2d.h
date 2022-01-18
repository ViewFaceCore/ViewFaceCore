#ifndef TENSORSTACK_KERNELS_CPU_GLOBAL_POOLING2D_H
#define TENSORSTACK_KERNELS_CPU_GLOBAL_POOLING2D_H

#include "operator_on_cpu.h"
#include "backend/base/base_global_pooling2d.h"
#include "pooling2d_core.h"


namespace ts {
    namespace cpu {
        using GlobalPooling2D = base::Pooling2DWithCore<OperatorOnCPU<base::GlobalPooling2D>, Pooling2DCore>;
    }
}


#endif //TENSORSTACK_KERNELS_CPU_GLOBAL_POOLING2D_H


