#include <backend/base/base_slice.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Slice::Slice() {
            field(name::begin,REQUIRED);
            field(name::size,REQUIRED);
        }

        void Slice::init() {
            supper::init();

            m_begin = tensor::array::to_int(get(name::begin));
            m_size = tensor::array::to_int(get(name::size));
        }

        static Tensor::Prototype infer_slice(const Tensor &x, const std::vector<int> &begins, const std::vector<int> & sizes) {
            Shape x_shape = x.sizes();

            TS_AUTO_CHECK(sizes.size() == begins.size());
            TS_AUTO_CHECK(sizes.size() == x_shape.size());

            Shape out_shape;
            out_shape.resize(x_shape.size());

            for(int i=0; i<begins.size(); i++) {
                TS_AUTO_CHECK((begins[i] >= 0) && (begins[i] < x_shape[i]));
                int dim = sizes[i];
                if(dim < 0) {
                    dim = x_shape[i] - begins[i];
                }

                TS_AUTO_CHECK(begins[i] + dim <= x_shape[i]);
                out_shape[i] = dim;
            }

            return Tensor::Prototype(x.dtype(), out_shape);
        }

        int Slice::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
             
            auto &x = stack[0];

            output.resize(1);
            output[0] = infer_slice(x, m_begin, m_size);

            return 1;
        }

        int Slice::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);

            auto output_proto = infer_slice(x, m_begin, m_size);
            auto &out = *stack.push(output_proto, memory_device);

            slice(x, out);

            return 1;
        }
    }

}
