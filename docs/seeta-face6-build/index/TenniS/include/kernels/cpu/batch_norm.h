#ifndef TENSORSTACK_KERNELS_CPU_BATCH_NORM_H
#define TENSORSTACK_KERNELS_CPU_BATCH_NORM_H

#include "operator_on_cpu.h"
#include "backend/base/base_batch_norm.h"


namespace ts {
    namespace cpu {
        class BatchNorm : public OperatorOnCPU<base::BatchNorm> {
        public:
            using self = BatchNorm;
            using supper = OperatorOnCPU<base::BatchNorm>;

            BatchNorm() = default;

            void batch_norm(const Tensor &x, const Tensor &mean, const Tensor &variance,
                            int dim, float epsilon, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_BATCH_NORM_H
