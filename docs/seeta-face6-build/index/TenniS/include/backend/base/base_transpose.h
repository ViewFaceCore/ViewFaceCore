//
// Created by kier on 2019/2/21.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_TRANSPOSE_H
#define TENSORSTACK_BACKEND_BASE_BASE_TRANSPOSE_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Transpose : public OperatorOnDevice {
        public:
            using self = Transpose;
            using supper = OperatorOnDevice;

            Transpose();

            void init() override;

            int run(ts::Stack &stack) override;

            int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) override;

            virtual void transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) = 0;

        private:
            std::vector<int> m_permute;

            std::vector<int> get_permute(const Tensor &x);
        };
    }
}

#endif  // TENSORSTACK_BACKEND_BASE_BASE_TRANSPOSE_H

