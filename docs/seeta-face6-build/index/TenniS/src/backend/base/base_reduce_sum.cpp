//
// Created by kier on 2019/7/23.
//

#include "backend/base/base_reduce_sum.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {
        ReduceSum::ReduceSum() {
            field(name::dims, REQUIRED);
            field(name::keep_dims, OPTIONAL, tensor::from<bool>(true));
        }

        void ReduceSum::init() {
            supper::init();

            m_dim = tensor::to_int(this->get(name::dims));
            m_keep_dim = tensor::to_bool(this->get(name::keep_dims));
        }

        static int checkout(Stack &stack, int dim, bool keep_dim, Shape &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            Shape input = stack[0].sizes();
            auto has_dims = int(input.size());

            int fixed_dim = dim >= 0 ? dim : has_dims + dim;

            if (fixed_dim < 0 || fixed_dim >= has_dims) {
                TS_LOG_ERROR << "Reduce dim must in [-"
                             << has_dims << ", "
                             << has_dims << ")" << eject;
            }

            if (keep_dim) {
                input[fixed_dim] = 1;
            } else {
                input.erase(input.begin() + fixed_dim);
            }

            output = std::move(input);

            return fixed_dim;
        }

        int ReduceSum::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            Shape output_shape;
            checkout(stack, m_dim, m_keep_dim, output_shape);

            output.resize(1);
            output[0] = Tensor::Prototype(stack[0].dtype(), output_shape);

            return 1;
        }

        int ReduceSum::run(Stack &stack) {
            Shape output_shape;
            auto fixed_dim = checkout(stack, m_dim, true, output_shape);

            Tensor::Prototype output_proto(stack[0].dtype(), output_shape);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto out = *stack.push(x.dtype(), output_shape, memory_device);

            reduce(x, fixed_dim, out);

            if (!m_keep_dim) {
                output_shape.erase(output_shape.begin() + fixed_dim);
                auto fixed_out = out.reshape(output_shape);
                stack.pop();
                stack.push(fixed_out);
            }

            return 1;
        }
    }
}
