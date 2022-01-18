//
// Created by kier on 2019/1/25.
//

#include "backend/zoo/divided.h"

#include "backend/name.h"

#include "core/tensor_builder.h"
#include "utils/assert.h"
#include "global/operator_factory.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"
#include "runtime/stack.h"

namespace ts {
    namespace zoo {
        Divided::Divided() {
            field(name::size, REQUIRED);
            field(name::padding_value, OPTIONAL);
        }

        void Divided::init() {
            supper::init();

            auto size = tensor::cast(INT32, this->get(name::size));

            m_size.clear();

            auto count = size.count();
            auto size_data = size.data<int32_t>();

            m_size.resize(count);
            for (int i = 0; i < count; ++i) {
                m_size[i] = size_data[i];
            }

            auto &context = ctx::ref<DeviceContext>();

            m_pad_op = OperatorCreator::Create(context.computing_device.type(), name::layer::pad(), false);

            TS_CHECK_NQ(m_pad_op, nullptr) << "Can not find operator: " << name::layer::pad();

            if (has(name::padding_value)) {
                m_pad_op->set(name::padding_value, get(name::padding_value).clone());
            }

            m_pad_op->init();

            m_padding = Tensor(INT32, {4, 2});
        }

        static inline void divided_shape(Shape &x, const std::vector<int32_t> &size) {
            if (size.size() > x.size()) {
                TS_LOG_ERROR << "Can not divided shape " << to_string(x) << " to " << to_string(size) << eject;
            }

            auto x_it = x.rbegin();
            auto size_it = size.rbegin();
            while (size_it != size.rend()) {
                auto stride = *size_it;
                auto &y = *x_it;
                y = stride == 1 ? y : int(std::ceil(float(y) / stride)) * stride;

                ++x_it;
                ++size_it;
            }
        }

        int Divided::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = *stack.index(0);
            auto shape = x.sizes();
            divided_shape(shape, m_size);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), std::move(shape));

            return 1;
        }

        int Divided::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = *stack.index(0);
            auto shape = x.sizes();
            divided_shape(shape, m_size);
            auto x_shape = x.sizes();

            if (m_padding.size(0) != shape.size()) {
                m_padding = Tensor(INT32, {int(shape.size()), 2});
            }

            auto padding = m_padding.data<int32_t>();

            for (size_t i = 0; i < shape.size(); ++i) {
                padding[i * 2] = 0;
                padding[i * 2 + 1] = shape[i] - x_shape[i];
            }

            stack.push(m_padding);

            TS_AUTO_CHECK(1 == RunOperator(m_pad_op, stack, 2));

            return 1;
        }
    }
}

using namespace ts;
using namespace zoo;

TS_REGISTER_OPERATOR(Divided, ts::CPU, ts::name::layer::divided());
