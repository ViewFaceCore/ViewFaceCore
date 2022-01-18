#include <backend/base/base_topkv2.h>

#include "backend/name.h"
#include "core/tensor_builder.h"
#include <algorithm>

namespace ts {
    namespace base {
        Topkv2::Topkv2() {
            field(name::number,REQUIRED);
            field(name::sorted,OPTIONAL, tensor::from<int32_t>(0));
        }

        void Topkv2::init() {
            supper::init();
            m_number = tensor::to_int(get(name::number));
            m_sorted = tensor::to_int(get(name::sorted));
        }


        int Topkv2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
             
            auto &x = stack[0];
            if (x.dims() == 0) {
                output = {x.proto(), {INT32, x.sizes()}};
                return 2;
            }

            Shape x_shape = x.sizes();
            auto K = std::min(x_shape.back(), m_number);
            x_shape.back() = K;

            output.resize(2);
            output[0] = Tensor::Prototype(x.dtype(), x_shape);
            output[1] = Tensor::Prototype(INT32, x_shape);

            return 2;
        }

        int Topkv2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = stack[0];

            if (x.dims() == 0) {
                auto values = x;
                int32_t i = 0;
                auto indices = tensor::build(INT32, {}, &i);
                stack.push(Tensor::Pack({values, indices}));
                return 1;
            }

            auto memory_device = running_memory_device();
            x = x.view(memory_device);

            Shape x_shape = x.sizes();
            auto K = std::min(x_shape.back(), m_number);
            x_shape.back() = K;

            auto values = stack.make(x.dtype(), x_shape, memory_device);
            auto indices = stack.make(INT32, x_shape, memory_device);

            topkv2(x, K, m_sorted, values, indices);

            stack.push(Tensor::Pack({values, indices}));

            return 1;
        }
    }

}
