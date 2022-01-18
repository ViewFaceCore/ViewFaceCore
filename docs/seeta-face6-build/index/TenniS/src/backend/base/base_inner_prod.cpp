//
// Created by kier on 2019/2/18.
//

#include <backend/base/base_inner_prod.h>
#include <backend/name.h>
#include <core/tensor_builder.h>

#include "backend/base/base_inner_prod.h"

namespace ts {
    namespace base {

        InnerProd::InnerProd() {
            field("transpose", OPTIONAL, tensor::from<bool>(false));
            field(name::kernel_packed, OPTIONAL, tensor::from<bool>(false));
        }

        void InnerProd::init() {
            supper::init();

            m_transpose = tensor::to_bool(get("transpose"));

            if (has(name::kernel_packed)) {
                m_kernel_packed = tensor::to_bool(get(name::kernel_packed));
            }
        }

        static void infer_size(bool m_transpose, Tensor &lhs, const Tensor &rhs, std::vector<Tensor::Prototype> &output) {
            if (lhs.dims() == 0) {
                TS_LOG_ERROR << "InnerProd failed with LHS is scalar." << eject;
            }

            if (lhs.dims() != 2) {
                lhs = lhs.reshape({lhs.size(0), -1});
            }

            if (m_transpose) {
                if (!(lhs.dims() == 2 && rhs.dims() == 2 && lhs.size(1) == rhs.size(1))) {
                    TS_LOG_ERROR << "Can not inner-product between "
                                 << to_string(lhs.sizes()) << " and "
                                 << to_string(rhs.sizes()) << "^T" << eject;
                }

                TS_AUTO_CHECK(lhs.dtype() == rhs.dtype());

                output.resize(1);
                output[0] = Tensor::Prototype(lhs.dtype(), {lhs.size(0), rhs.size(0)});
            } else {
                if (!(lhs.dims() == 2 && rhs.dims() == 2 && lhs.size(1) == rhs.size(0))) {
                    TS_LOG_ERROR << "Can not inner-product between "
                                 << to_string(lhs.sizes()) << " and "
                                 << to_string(rhs.sizes()) << eject;
                }

                TS_AUTO_CHECK(lhs.dtype() == rhs.dtype());

                output.resize(1);
                output[0] = Tensor::Prototype(lhs.dtype(), {lhs.size(0), rhs.size(1)});
            }
        }

        int InnerProd::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto lhs = stack[0];
            auto &rhs = stack[1];

            infer_size(m_transpose, lhs, rhs, output);

            return 1;
        }

        int InnerProd::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            std::vector<Tensor::Prototype> output;
            auto lhs = stack[0];
            auto rhs = stack[1];

            infer_size(m_transpose, lhs, rhs, output);

            auto memory_device = running_memory_device();

            lhs = lhs.view(memory_device);
            rhs = rhs.view(memory_device);

            auto &out = *stack.push(output[0], memory_device);

            inner_prod(lhs, rhs, m_transpose, out, stack, m_kernel_packed);

            return 1;
        }
    }
}