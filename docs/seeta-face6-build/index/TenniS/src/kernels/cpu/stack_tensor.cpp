#include <kernels/cpu/stack_tensor.h>
#include <core/tensor_builder.h>

#include "backend/name.h"
#include "core/memory.h"
#include "global/operator_factory.h"
#include "global/hard_converter.h"

#include "core/device_context.h"
#include "utils/ctxmgr.h"

namespace ts {
    namespace cpu {
        void StackTensor::stack_tensor(const std::vector<ts::Tensor> &x, int axis, ts::Tensor &out) {
            throw Exception("What a Terrible Failure!");
        }

        int StackTensor::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() > 0);

            int output_dims = int(stack[0].dims() + 1);
            int fixed_axis = m_axis >= 0 ? m_axis : output_dims + m_axis;

            if (fixed_axis < 0 || fixed_axis >= output_dims) {
                TS_LOG_ERROR << "Stack axis must in [-"
                             << output_dims << ", "
                             << output_dims << ")" << eject;
            }

            int nargs = int(stack.size());
            for (int i = 0; i < nargs; ++i) {
                auto &x = stack[i];
                auto shape = x.sizes();
                shape.insert(shape.begin() + fixed_axis, 1);
                stack.push(x.reshape(shape));
            }
            return RunOperator(m_op_concat, stack, nargs);
        }

        void StackTensor::init() {
            supper::init();

            auto &context = ctx::ref<DeviceContext>();

            m_op_concat = OperatorCreator::Create(context.computing_device.type(), name::layer::concat(), false);
            m_op_concat->set(name::dim, tensor::from<int>(m_axis));
            m_op_concat->init();
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(StackTensor, ts::CPU, name::layer::stack())