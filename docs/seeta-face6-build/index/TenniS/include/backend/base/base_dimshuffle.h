//
// Created by kier on 2019/2/17.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_DIMSHUFFLE_H
#define TENSORSTACK_BACKEND_BASE_BASE_DIMSHUFFLE_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Dimshuffle : public OperatorOnDevice {
        public:
            using self = Dimshuffle;
            using supper = OperatorOnDevice;

            Dimshuffle();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void dimshuffle(const Tensor &x, int dim, const std::vector<int> &shuffle, Tensor &out) = 0;

        private:
            int m_dim = -1;
            std::vector<int> m_shuffle;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_DIMSHUFFLE_H
