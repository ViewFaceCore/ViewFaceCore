//
// Created by kier on 2019/2/18.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_PAD_H
#define TENSORSTACK_BACKEND_BASE_BASE_PAD_H

#include "operator_on_device.h"
#include <array>

namespace ts {
    namespace base {
        class Pad : public OperatorOnDevice {
        public:
            using self = Pad;
            using supper = OperatorOnDevice;

            Pad();

            void init() override;

            /**
             *
             * @param stack x[1...N], padding[N, 2]
             * @return
             */
            int run(ts::Stack &stack) override;

            int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) override;

            virtual void
            pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) = 0;

        private:
            float m_padding_value;

        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_PAD_H
