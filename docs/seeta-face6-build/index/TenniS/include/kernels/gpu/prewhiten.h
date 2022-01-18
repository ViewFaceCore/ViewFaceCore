#ifndef TENSORSTACK_KERNELS_GPU_PREWHITEN_H
#define TENSORSTACK_KERNELS_GPU_PREWHITEN_H

#include "backend/base/base_prewhiten.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class PreWhiten : public OperatorOnGPU<base::PreWhiten> {
        public:
            using self = PreWhiten;
            using supper = OperatorOnGPU<base::PreWhiten>;

            void prewhiten(const Tensor &x, Tensor &out) override;
        };
    }
}



#endif //TENSORSTACK_KERNELS_GPU_PREWHITEN_H