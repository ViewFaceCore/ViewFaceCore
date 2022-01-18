#include <backend/base/base_max.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Max::Max() {
            field(name::dim, REQUIRED);
            field(name::keep_dims, OPTIONAL);
            m_keep_dims = 1;
        }

        void Max::init() {
            supper::init();
             
            Tensor dim_tensor = tensor::cast(INT32, get(name::dim));
            m_dim = tensor::to_int(dim_tensor);
            if(has(name::keep_dims)) {
                Tensor keep_dims_tensor = tensor::cast(INT32, get(name::keep_dims));
                m_keep_dims = tensor::to_int(keep_dims_tensor);
            }

        }

        static Tensor::Prototype infer_max(const Tensor &x, int dim, int keep_dims) {
            Shape x_shape = x.sizes();
            if(dim < 0) {
               dim += int(x_shape.size());
            }

            TS_AUTO_CHECK((dim >= 0) && (dim < int(x_shape.size())));

            if(keep_dims == 1) {
                x_shape[dim] = 1;
            }else {
                x_shape.erase(x_shape.begin() + dim);
            }

            return Tensor::Prototype(x.dtype(), x_shape);
        }

        int Max::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto &x = stack[0];
            output.resize(1);
            output[0] = infer_max(x, m_dim, m_keep_dims);

            return 1;
        }

        int Max::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);

            auto output_proto = infer_max(x, m_dim, m_keep_dims);
            auto &out = *stack.push(output_proto, memory_device);

            max(x, out);

            return 1;
        }
    }

}
