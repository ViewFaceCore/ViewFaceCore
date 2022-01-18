#ifndef TENSORSTACK_KERNELS_GPU_GEMM_H
#define TENSORSTACK_KERNELS_GPU_GEMM_H

#include "operator_on_gpu.h"
#include "backend/base/base_gemm.h"


namespace ts {
    namespace gpu {
        class Gemm : public OperatorOnGPU<base::Gemm> {
        public:
            using self = Gemm;
            using supper = OperatorOnGPU<base::Gemm>;

            void gemm(const Tensor &A, const Tensor &B, const Tensor &C, int K,
                      float alpha, float beta, bool transA, bool transB, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_GEMM_H
