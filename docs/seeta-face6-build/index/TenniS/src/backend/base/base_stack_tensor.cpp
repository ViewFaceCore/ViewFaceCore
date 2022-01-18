//
// Created by kier on 2019/4/9.
//

#include <backend/base/base_stack_tensor.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {

        StackTensor::StackTensor() {
            field(name::axis, OPTIONAL, tensor::from<int>(0));
        }

        void StackTensor::init() {
            supper::init();

            m_axis = tensor::to_int(this->get(name::axis));

            // TS_AUTO_CHECK(m_axis >= 0);
        }

        int StackTensor::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto input_num = stack.size();

            TS_AUTO_CHECK(input_num != 0);
            // TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

            auto &x = stack[0];
            for (int i = 1; i < input_num; ++i) {
                TS_AUTO_CHECK(stack[i].has_shape(x.sizes()) && stack[i].dtype() == x.dtype());
            }

            auto shape = x.sizes();
            int output_dims = int(shape.size() + 1);
            int fixed_axis = m_axis >= 0 ? m_axis : output_dims + m_axis;

            if (fixed_axis < 0 || fixed_axis >= output_dims) {
                TS_LOG_ERROR << "Stack axis must in [-"
                             << output_dims << ", "
                             << output_dims << ")" << eject;
            }

            shape.insert(shape.begin() + fixed_axis, int(stack.size()));

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), shape);

            return 1;
        }

        int StackTensor::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto input_num = stack.size();

            auto memory_device = running_memory_device();

            std::vector<Tensor> x;
            for (size_t i = 0; i < input_num; ++i) {
                x.emplace_back(stack[i].view(memory_device));
            }

            Tensor out = *stack.push(output_protos[0], memory_device);

            int output_dims = int(x[0].dims() + 1);
            int fixed_axis = m_axis >= 0 ? m_axis : output_dims + m_axis;

            if (fixed_axis < 0 || fixed_axis >= output_dims) {
                TS_LOG_ERROR << "Stack axis must in [-"
                             << output_dims << ", "
                             << output_dims << ")" << eject;
            }

            stack_tensor(x, fixed_axis, out);

            return 1;
        }
    }
}
