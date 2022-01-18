#ifndef TENSORSTACK_BACKEND_BASE_BASE_TOPKV2_H
#define TENSORSTACK_BACKEND_BASE_BASE_TOPKV2_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Topkv2 : public OperatorOnDevice {
        public:
            using self = Topkv2;
            using supper = OperatorOnDevice;

            Topkv2();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param out
             */
            virtual void topkv2(const Tensor &x, int K, bool sorted, Tensor &values, Tensor &indices) = 0;

        protected:
            int m_number;
            int m_sorted;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_TOPKV2_H
