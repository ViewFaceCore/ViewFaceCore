#include <backend/base/base_winograd_transform_kernel.h>

#include "backend/base/base_winograd_transform_kernel.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        WinogradTransKernel::WinogradTransKernel() {
            field(name::winograd_mode,OPTIONAL,tensor::from(name::winograd_f63));
        }

        void WinogradTransKernel::init() {
            supper::init();

            auto winograd_model = tensor::to_string(get(name::winograd_mode));

            if (winograd_model == name::winograd_f63) {
                m_winograd_mode = F6X6_3X3;
            }
            else if (winograd_model == name::winograd_f23) {
                m_winograd_mode = F2X2_3X3;
            }
            else {
                TS_LOG_ERROR << this->op() << " do not support winograd model: " << winograd_model << eject;
            }
        }

        int WinogradTransKernel::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto kernel_tensor = stack[0];
            TS_AUTO_CHECK(kernel_tensor.dims() == 4);

            auto kernel_shape = kernel_tensor.sizes();
            TS_AUTO_CHECK(kernel_shape[2] == 3 && kernel_shape[3] == 3);

            Shape output_shape = kernel_shape;
            if (m_winograd_mode == F6X6_3X3) {
                output_shape[2] = 8;
                output_shape[3] = 8;
            }
            else if (m_winograd_mode == F2X2_3X3) {
                output_shape[2] = 4;
                output_shape[3] = 4;
            }

            output.resize(1);
            output[0] = Tensor::Prototype(kernel_tensor.dtype(), output_shape);

            return 1;
        }

        int WinogradTransKernel::run(ts::Stack &stack) {
            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto memory_device = running_memory_device();

            auto kernel_tensor = stack[0].view(memory_device);

            auto out = *stack.push(output[0], memory_device);

            transform_kernel(kernel_tensor, m_winograd_mode, out);

            return 1;
        }
    }
}