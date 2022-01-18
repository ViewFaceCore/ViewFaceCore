#ifndef TENSORSTACK_KERNELS_CPU_NON_MAX_SUPPRESSION_V3_H
#define TENSORSTACK_KERNELS_CPU_NON_MAX_SUPPRESSION_V3_H

#include "operator_on_cpu.h"
#include "backend/base/base_non_max_suppression_v3.h"


namespace ts {
    namespace cpu {
        class Non_Max_Suppression_V3 : public OperatorOnCPU<base::Non_Max_Suppression_V3> {
        public:
            using self = Non_Max_Suppression_V3;
            using supper = OperatorOnCPU<base::Non_Max_Suppression_V3>;

            void non_max_suppression_v3(const Tensor &x, const Tensor &scores, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_NON_MAX_SUPPRESSION_V3_H
