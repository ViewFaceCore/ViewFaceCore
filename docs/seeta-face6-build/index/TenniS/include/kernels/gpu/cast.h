
#ifndef TENSORSTACK_BACKEND_kernels_CAST_V2_H
#define TENSORSTACK_BACKEND_kernels_CAST_V2_H

#include "backend/base/base_cast_v2.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class CastV2 : public OperatorOnGPU<base::CastV2> {
        public:
            using self = CastV2;
            using supper = OperatorOnGPU<base::CastV2>;

            virtual void cast(const Tensor &x, DTYPE dtype, Tensor &out) override;

        };

    }
}


#endif //TENSORSTACK_BACKEND_kernels_CAST_V2_H

