//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_FUSED_BATCH_NORM_H
#define TENSORSTACK_BACKEND_BASE_BASE_FUSED_BATCH_NORM_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        class FusedBatchNorm : public OperatorOnDevice {
        public:
            using self = FusedBatchNorm;
            using supper = OperatorOnDevice;

            FusedBatchNorm();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void batch_norm(const Tensor &x,
                    const Tensor &mean, const Tensor &variance,
                    const Tensor &scale, const Tensor &bias,
                    int dim, float epsilon, Tensor &out) = 0;

        private:
            float m_epsilon = 1e-5f;
            int m_dim = -1;

            bool check_inputs(Stack &stack) const;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_FUSED_BATCH_NORM_H
