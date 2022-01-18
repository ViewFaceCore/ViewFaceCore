#ifndef TENSORSTACK_KERNELS_GPU_FUSED_BATCH_NORM_H
#define TENSORSTACK_KERNELS_GPU_FUSED_BATCH_NORM_H

#include "operator_on_gpu.h"
#include "backend/base/base_fused_batch_norm.h"


namespace ts {
    namespace gpu {
        class FusedBatchNorm : public OperatorOnGPU<base::FusedBatchNorm> {
        public:
            using self = FusedBatchNorm;
            using supper = OperatorOnGPU<base::FusedBatchNorm>;

            FusedBatchNorm() = default;

            void batch_norm(const Tensor &x, const Tensor &mean, const Tensor &variance,
                            const Tensor &scale, const Tensor &bias,
                            int dim, float epsilon, Tensor &out) override;
        };
    }
}

#endif
