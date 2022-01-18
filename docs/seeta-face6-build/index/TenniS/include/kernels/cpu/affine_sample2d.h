#ifndef TENSORSTACK_KERNELS_CPU_AFFINE_SAMPLE2D_H
#define TENSORSTACK_KERNELS_CPU_AFFINE_SAMPLE2D_H

#include "operator_on_cpu.h"
#include "backend/base/base_affine_sample2d.h"


namespace ts {
    namespace cpu {
        class Affine_Sample2D : public OperatorOnCPU<base::Affine_Sample2D> {
        public:
            using self = Affine_Sample2D;
            using supper = OperatorOnCPU<base::Affine_Sample2D>;

            Affine_Sample2D() = default;

            void affine_sample_run(const Tensor &x, float rz00, float rz01, float rz02, float rz10, float rz11, float rz12,
                                           float rz20, float rz21, float rz22, Affine_Sample2DType type, int dim,
                                           base::AffineOuterMode outer_mode, float outer_value,
                                           Tensor &out) override;
        };
    }
}

#endif  
