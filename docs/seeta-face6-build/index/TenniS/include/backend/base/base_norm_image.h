//
// Created by kier on 2019/6/29.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_NORM_IMAGE_H
#define TENSORSTACK_BACKEND_BASE_BASE_NORM_IMAGE_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class NormImage : public OperatorOnDevice {
        public:
            using self = NormImage;
            using supper = OperatorOnDevice;

            NormImage();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void norm_image(const Tensor &x, float epsilon, Tensor &out) = 0;

        public:
            float m_epsilon = 1e-5f;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_NORM_IMAGE_H
