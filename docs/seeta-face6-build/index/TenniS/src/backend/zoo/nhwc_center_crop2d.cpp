//
// Created by kier on 2019/1/25.
//

#include "backend/zoo/nhwc_center_crop2d.h"

#include "backend/name.h"

#include "core/tensor_builder.h"
#include "utils/assert.h"
#include "global/operator_factory.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"
#include "runtime/stack.h"

namespace ts {
    namespace zoo {

        NHWCCenterCrop2D::NHWCCenterCrop2D() {
            field(name::size, REQUIRED);
        }

        void NHWCCenterCrop2D::init() {
            supper::init();

            auto size = tensor::cast(INT32, this->get(name::size));

            TS_AUTO_CHECK(size.has_shape({2}));

            m_size.width = size.data<int32_t>(0);
            m_size.height = size.data<int32_t>(1);

            auto &context = ctx::ref<DeviceContext>();

            m_pad_op = OperatorCreator::Create(context.computing_device.type(), name::layer::pad(), false);

            TS_CHECK_NQ(m_pad_op, nullptr) << "Can not find operator: " << name::layer::pad();

            m_pad_op->set(name::padding_value, tensor::build(FLOAT32, {0.0f}));

            m_pad_op->init();
        }

        int NHWCCenterCrop2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = *stack.index(0);

            TS_AUTO_CHECK(x.dims() == 4);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), {x.size(0), m_size.height, m_size.width, x.size(3)});

            return 1;
        }

        int NHWCCenterCrop2D::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto x = *stack.index(0);

            TS_AUTO_CHECK(x.dims() == 4);

            auto x_size = Size2D(x.size(1), x.size(2));

            auto x_padding_top = (m_size.height - x_size.height) / 2;
            auto x_padding_bottom = m_size.height - x_size.height - x_padding_top;

            auto x_padding_left = (m_size.width - x_size.width) / 2;
            auto x_padding_right = m_size.width - x_size.width - x_padding_left;

            auto x_padding_tensor = tensor::build(INT32, {4, 2}, {
                    0, 0,
                    x_padding_top, x_padding_bottom,
                    x_padding_left, x_padding_right,
                    0, 0,
            });

            stack.push(x_padding_tensor);

            TS_AUTO_CHECK(1 == RunOperator(m_pad_op, stack, 2));

            return 1;
        }
    }
}

using namespace ts;
using namespace zoo;

TS_REGISTER_OPERATOR(NHWCCenterCrop2D, ts::CPU, ts::name::layer::nhwc_center_crop2d());
