//
// Created by kier on 2019/2/20.
//

#include "backend/base/base_cast_v2.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {

        CastV2::CastV2() {
            field(name::dtype, REQUIRED);
        }

        void CastV2::set_dtype(DTYPE dtype) {
            m_dtype = dtype;
        }

        DTYPE CastV2::get_dtype() const {
            return m_dtype;
        }

        void CastV2::init() {
            m_dtype = DTYPE(tensor::to_int(get(name::dtype)));
        }

        int CastV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            output.resize(1);
            output[0] = Tensor::Prototype(m_dtype, x.sizes());

            return 1;
        }

        int CastV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            if (x.dtype() == m_dtype) return 1;

            auto out = *stack.push(m_dtype, x.sizes(), memory_device);

            cast(x, m_dtype, out);

            return 1;
        }

        void CastV2::cast(const Tensor &x, DTYPE dtype, Tensor &out) {
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
