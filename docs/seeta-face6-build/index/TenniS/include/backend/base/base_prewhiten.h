//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_PREWHITEN_H
#define TENSORSTACK_BACKEND_BASE_BASE_PREWHITEN_H

#include "base_activation.h"

namespace ts {
    namespace base {
        class PreWhiten : public Activation {
        public:
            using self = PreWhiten;
            using supper = Activation;

            void active(const Tensor &x, Tensor &out) final;

            /**
             * calculation
             * @param x satisfied shape.dims() > 1
             * @param out have same shape with x
             * @note all Tensor parameters' memory are already sync to given running memory device.
             */
            virtual void prewhiten(const Tensor &x, Tensor &out) = 0;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_PREWHITEN_H
