//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_BATCH_SCLAE_H
#define TENSORSTACK_BACKEND_BASE_BASE_BATCH_SCLAE_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         * AddBias
         */
        class BatchScale : public OperatorOnDevice {
        public:
            using self = BatchScale;
            using supper = OperatorOnDevice;

            BatchScale();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void batch_scale(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                     int dim, Tensor &out) = 0;

        private:
            int m_dim = -1;

            bool check_inputs(Stack &stack) const;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_BATCH_SCLAE_H
