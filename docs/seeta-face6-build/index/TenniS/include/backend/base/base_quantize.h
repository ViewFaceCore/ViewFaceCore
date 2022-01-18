#ifndef TENSORSTACK_BACKEND_BASE_QUANTIZE_H
#define TENSORSTACK_BACKEND_BASE_QUANTIZE_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        /**
        * Quantize
        */
        class Quantize : public OperatorOnDevice {
        public:
            using self = Quantize;
            using supper = OperatorOnDevice;

            Quantize();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void quantize(const Tensor &x, std::vector<float> quantize_scales, Tensor &out) = 0;

        private:
            std::vector<float> m_quantize_scales;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_QUANTIZE_H
