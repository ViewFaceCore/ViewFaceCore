//
// Created by kier on 2019/3/6.
//

#ifndef TENSORSTACK_KERNELS_GPU_GATHERV2_H
#define TENSORSTACK_KERNELS_GPU_GATHERV2_H

#include "operator_on_gpu.h"
#include "backend/base/base_gatherv2.h"


namespace ts {
    namespace gpu {
        class GatherV2 : public OperatorOnAny<base::GatherV2> {
        public:
            using self = GatherV2;
            using supper = OperatorOnGPU<base::GatherV2>;

            void gather(const Tensor &x, const Tensor &indices, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_GATHERV2_H
