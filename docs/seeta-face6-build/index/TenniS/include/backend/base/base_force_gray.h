//
// Created by kier on 2019/6/26.
//

#ifndef TENSORSTACK_BACKEND_BASE_FORCE_GRAY_H
#define TENSORSTACK_BACKEND_BASE_FORCE_GRAY_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         *
         */
        class ForceGray : public OperatorOnDevice {
        public:
            using self = ForceGray;
            using supper = OperatorOnDevice;

            ForceGray();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void force_gray(const Tensor &x, const std::vector<float> &scale, Tensor &out) = 0;

        private:
            std::vector<float> m_scale;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_FORCE_GRAY_H
