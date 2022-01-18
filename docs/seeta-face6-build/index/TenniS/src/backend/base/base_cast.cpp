//
// Created by kier on 2019/2/20.
//

#include <backend/base/base_cast.h>

#include "backend/base/base_cast.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {

        Cast::Cast(DTYPE dtype)
                : m_dtype(dtype) {}

        void Cast::set_dtype(DTYPE dtype) {
            m_dtype = dtype;
        }

        int Cast::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            output.resize(1);
            output[0] = Tensor::Prototype(m_dtype, x.sizes());

            return 1;
        }

        int Cast::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            if (x.dtype() == m_dtype) return 1;

            auto out = *stack.push(m_dtype, x.sizes(), memory_device);

            cast(x, m_dtype, out);

            return 1;
        }

        void Cast::cast(const Tensor &x, DTYPE dtype, Tensor &out) {
            if (x.dtype() == dtype) {
                auto src = x.weak_memory();
                auto dst = out.weak_memory();
                memcpy(dst, src);
                return;
            }
            auto temp = tensor::cast(dtype, x);
            auto src = temp.weak_memory();
            auto dst = out.weak_memory();
            memcpy(dst, src);
        }
    }
}
