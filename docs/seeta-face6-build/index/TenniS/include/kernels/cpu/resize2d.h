#ifndef TENSORSTACK_KERNELS_CPU_RESIZE2D_H
#define TENSORSTACK_KERNELS_CPU_RESIZE2D_H

#include "backend/base/base_resize2d.h"
#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class Resize2D : public OperatorOnCPU<base::Resize2D> {
        public:
            void resize2d(const Tensor &x, int dim, Resize2DType type, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_RESIZE2D_H
