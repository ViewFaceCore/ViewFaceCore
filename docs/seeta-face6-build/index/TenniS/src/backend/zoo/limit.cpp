//
// Created by kier on 2019/3/12.
//

#include "backend/zoo/limit.h"

#include "backend/name.h"

#include "core/tensor_builder.h"
#include "utils/assert.h"
#include "global/operator_factory.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"
#include "runtime/stack.h"

namespace ts {
    namespace zoo {

        Limit::Limit() {
            field(name::shape, REQUIRED);
        }

        void Limit::init() {
            supper::init();

            auto shape = tensor::cast(INT32, this->get(name::shape));
            TS_AUTO_CHECK(shape.dims() == 1);
            auto shape_data = shape.data<int32_t>();

            m_shape.resize(shape.size(0));
            for (size_t i = 0; i < m_shape.size(); ++i) {
                m_shape[i] = shape_data[i];
            }

            auto &context = ctx::ref<DeviceContext>();

            m_pad_op = OperatorCreator::Create(context.computing_device.type(), name::layer::pad(), false);

            TS_CHECK_NQ(m_pad_op, nullptr) << "Can not find operator: " << name::layer::pad();

            m_pad_op->set(name::padding_value, tensor::build(FLOAT32, {0.0f}));

            m_pad_op->init();
        }

        static inline bool if_overflow(const Shape &x_shape, std::vector<int> &limit) {
            TS_AUTO_CHECK(x_shape.size() >= limit.size());

            if (limit.size() < x_shape.size()) {
                std::vector<int> ones(x_shape.size() - limit.size(), -1);
                limit.insert(limit.begin(), ones.begin(), ones.end());
            }

            auto size = x_shape.size();

            bool overflowed = false;
            for (size_t i = 0; i < size; ++i) {
                if (limit[i] <= 0 || x_shape[i] <= limit[i]) {
                    limit[i] = x_shape[i];
                } else {
                    overflowed = true;
                }
            }
            return overflowed;
        }

        int Limit::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = *stack.index(0);

            std::vector<int> limit = m_shape;

            if_overflow(x.sizes(), limit);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), std::move(limit));

            return 1;
        }

        int Limit::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = *stack.index(0);
            auto &x_shape = x.sizes();

            std::vector<int> limit = m_shape;

            auto overflowed = if_overflow(x.sizes(), limit);
            if (!overflowed) return 1;

            auto x_padding_tensor = Tensor(INT32, Shape({int(limit.size()), 2}));
            auto x_padding_data = x_padding_tensor.data<int32_t>();
            auto limit_count = limit.size();
            for (size_t i = 0; i < limit_count; ++i) {
                auto neg_padding = limit[i] - x_shape[i];
                auto half_neg_padding = neg_padding / 2;
                x_padding_data[i * 2] = half_neg_padding;
                x_padding_data[i * 2 + 1] = neg_padding - half_neg_padding;
            }

            stack.push(x_padding_tensor);

            TS_AUTO_CHECK(1 == RunOperator(m_pad_op, stack, 2));

            return 1;
        }
    }
}

using namespace ts;
using namespace zoo;

TS_REGISTER_OPERATOR(Limit, ts::CPU, ts::name::layer::limit());
