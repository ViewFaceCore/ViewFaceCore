#ifndef TENSORSTACK_KERNELS_GPU_SPACETOBATCH4D_H
#define TENSORSTACK_KERNELS_GPU_SPACETOBATCH4D_H

#include "operator_on_gpu.h"
#include "backend/base/base_spacetobatch4d.h"


namespace ts {
    namespace gpu {
        class SpaceToBatch4D : public OperatorOnGPU<base::SpaceToBatch4D> {
        public:
            using self = SpaceToBatch4D;
            using supper = OperatorOnGPU<base::SpaceToBatch4D>;

            SpaceToBatch4D() = default;

            void spacetobatch4d_run(const Tensor &x,const int padding_top, const int padding_bottom,
                    const int padding_left,const int padding_right, const int block_height, const int block_width, Tensor &out) override;
        };
    }
}

#endif  
