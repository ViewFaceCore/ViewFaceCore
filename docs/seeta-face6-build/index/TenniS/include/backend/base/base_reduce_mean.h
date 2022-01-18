//
// Created by kier on 2019/7/23.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_REDUCE_MEAN_H
#define TENSORSTACK_BACKEND_BASE_BASE_REDUCE_MEAN_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         *
         */
        class ReduceMean : public OperatorOnDevice {
        public:
            using self = ReduceMean;
            using supper = OperatorOnDevice;

            ReduceMean();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param dim reduce dim
             * @param out output tensor with kept dim
             */
            virtual void reduce(const Tensor &x, std::vector<int> dims, Tensor &out) = 0;

        private:
//            int m_dim = -1;
            std::vector<int> m_dims;
            bool m_keep_dim = true;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_REDUCE_MEAN_H
