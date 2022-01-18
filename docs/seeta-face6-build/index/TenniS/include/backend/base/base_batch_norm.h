//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_BATCH_NORM_H
#define TENSORSTACK_BACKEND_BASE_BASE_BATCH_NORM_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         * AddBias
         */
        class BatchNorm : public OperatorOnDevice {
        public:
            using self = BatchNorm;
            using supper = OperatorOnDevice;

            BatchNorm();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void batch_norm(const Tensor &x, const Tensor &mean, const Tensor &variance,
                    int dim, float epsilon, Tensor &out) = 0;

        private:
            float m_epsilon = 1e-5f;
            int m_dim = -1;

            bool check_inputs(Stack &stack) const;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_BATCH_NORM_H
