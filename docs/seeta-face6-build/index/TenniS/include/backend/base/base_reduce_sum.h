//
// Created by kier on 2019/7/23.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_REDUCE_SUM_H
#define TENSORSTACK_BACKEND_BASE_BASE_REDUCE_SUM_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         *
         */
        class ReduceSum : public OperatorOnDevice {
        public:
            using self = ReduceSum;
            using supper = OperatorOnDevice;

            ReduceSum();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param dim reduce dim
             * @param out output tensor with kept dim
             */
            virtual void reduce(const Tensor &x, int dim, Tensor &out) = 0;

        private:
            int m_dim = -1;
            bool m_keep_dim = true;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_REDUCE_SUM_H
