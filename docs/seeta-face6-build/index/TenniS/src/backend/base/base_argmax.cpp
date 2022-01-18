#include <backend/base/base_argmax.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        ArgMax::ArgMax() {
            field(name::dim, REQUIRED);
        }

        void ArgMax::init() {
            supper::init();
             
            Tensor dim_tensor = tensor::cast(INT32, get(name::dim));
            m_dim = tensor::to_int(dim_tensor);

        }

        static Tensor::Prototype infer_argmax(const Tensor &x, int dim) {
            Shape x_shape = x.sizes();
            if(dim < 0) {
               dim += int(x_shape.size());
            }

            TS_AUTO_CHECK((dim >= 0) && (dim < int(x_shape.size())));

            x_shape.erase(x_shape.begin() + dim);

            return Tensor::Prototype(INT32, x_shape);
        }

        int ArgMax::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto &x = stack[0];
            output.resize(1);
            output[0] = infer_argmax(x, m_dim);

            return 1;
        }

        int ArgMax::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);

            auto output_proto = infer_argmax(x, m_dim);
            auto &out = *stack.push(output_proto, memory_device);


            argmax(x, m_dim, out);

            return 1;
        }
    }

}
