#ifndef TENSORSTACK_KERNELS_GPU_BATCHTOSPACE4D_H
#define TENSORSTACK_KERNELS_GPU_BATCHTOSPACE4D_H

#include "operator_on_gpu.h"
#include "backend/base/base_batchtospace4d.h"


namespace ts {
    namespace gpu {
        class BatchToSpace4D : public OperatorOnGPU<base::BatchToSpace4D> {
        public:
            using self = BatchToSpace4D;
            using supper = OperatorOnGPU<base::BatchToSpace4D>;

            BatchToSpace4D() = default;

            void batchtospace4d_run(const Tensor &x,const int crop_top, const int crop_bottom,
                    const int crop_left,const int crop_right, const int block_height, const int block_width, Tensor &out) override;
        };
    }
}

#endif  
