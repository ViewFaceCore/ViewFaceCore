#ifndef TENSORSTACK_KERNELS_CPU_SHAPE_INDEX_PATCH_H
#define TENSORSTACK_KERNELS_CPU_SHAPE_INDEX_PATCH_H

#include "backend/base/base_shape_index_patch.h"
#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class ShapeIndexPatch : public OperatorOnCPU<base::ShapeIndexPatch> {
        public:
            using self = ShapeIndexPatch;

            void sample(const Tensor &x, const Tensor &pos, const Size2D &origin_patch, const Size2D &origin, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_SHAPE_INDEX_PATCH_H
