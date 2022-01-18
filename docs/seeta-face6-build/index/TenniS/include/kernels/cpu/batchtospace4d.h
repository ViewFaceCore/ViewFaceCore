#ifndef TENSORSTACK_KERNELS_CPU_BATCHTOSPACE4D_H
#define TENSORSTACK_KERNELS_CPU_BATCHTOSPACE4D_H

#include "operator_on_cpu.h"
#include "backend/base/base_batchtospace4d.h"


namespace ts {
    namespace cpu {
        class BatchToSpace4D : public OperatorOnCPU<base::BatchToSpace4D> {
        public:
            using self = BatchToSpace4D;
            using supper = OperatorOnCPU<base::BatchToSpace4D>;

            BatchToSpace4D() = default;

            void batchtospace4d_run(const Tensor &x,const int crop_top, const int crop_bottom,
                    const int crop_left,const int crop_right, const int block_height, const int block_width, Tensor &out) override;
        };
    }
}

#endif  
