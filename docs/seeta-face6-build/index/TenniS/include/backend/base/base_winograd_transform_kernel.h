#ifndef TENSORSTACK_BACKEND_BASE_BASE_WINOGRAD_TRANSFORM_KERNEL_H
#define TENSORSTACK_BACKEND_BASE_BASE_WINOGRAD_TRANSFORM_KERNEL_H

#include "operator_on_device.h"
#include "backend/common_structure.h"

namespace ts {
    namespace base {
        class WinogradTransKernel : public OperatorOnDevice {
        public:
            using self = WinogradTransKernel;
            using supper = OperatorOnDevice;

            WinogradTransKernel();
            
            void init() override;

            int run(ts::Stack &stack) override;

            int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) override;

            virtual void transform_kernel(const Tensor &x, WinogradConv2DMode winograd_mode, Tensor &out) = 0;

        private:
            WinogradConv2DMode m_winograd_mode;
        };
    }
}

#endif