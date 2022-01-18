//
// Created by kier on 2019/6/26.
//

#ifndef TENSORSTACK_BACKEND_BASE_FORCE_COLOR_H
#define TENSORSTACK_BACKEND_BASE_FORCE_COLOR_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
         *
         */
        class ForceColor : public OperatorOnDevice {
        public:
            using self = ForceColor;
            using supper = OperatorOnDevice;

            ForceColor();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x channel = 1
             * @param out
             */
            virtual void force_color(const Tensor &x, Tensor &out) = 0;

        private:
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_FORCE_COLOR_H
