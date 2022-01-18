//
// Created by kier on 2019/4/9.
//

#include "backend/zoo/crop_nd.h"

#include "backend/name.h"

#include "core/tensor_builder.h"
#include "utils/assert.h"
#include "global/operator_factory.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"
#include "runtime/stack.h"

#include <array>

namespace ts {
    namespace zoo {
        CropND::CropND() {
            field(name::shift, OPTIONAL);
        }

        void CropND::init() {
            supper::init();

            m_shift.clear();
            if (has(name::shift)) {
                m_shift = tensor::array::to_int(this->get(name::shift));
            }

            auto &context = ctx::ref<DeviceContext>();

            m_pad_op = OperatorCreator::Create(context.computing_device.type(), name::layer::pad(), false);

            TS_CHECK_NQ(m_pad_op, nullptr) << "Can not find operator: " << name::layer::pad();

            m_pad_op->set(name::padding_value, tensor::build(FLOAT32, {0.0f}));

            m_pad_op->init();
        }

        int CropND::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto &size = stack[1];
            auto shape = tensor::array::to_int(size);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), shape);

            return 1;
        }

        int CropND::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x = *stack.index(0);
            auto shape = tensor::cast(INT32, stack[1]);
            auto shape_count = shape.count();
            auto shape_data = shape.data<int32_t>();
            auto &x_size = x.sizes();

            auto dims = x.dims();

            TS_AUTO_CHECK((m_shift.empty() || m_shift.size() == dims) && shape_count == dims);

            std::vector<std::array<int32_t, 2>> padding(dims);

            if (m_shift.empty()) {
                for (int i = 0; i < dims; ++i) {
                    if (shape_data[i] <= 0) {
                        padding[i][0] = 0;
                        padding[i][1] = 0;
                    } else {
                        auto x_padding_top = (shape_data[i] - x_size[i]) / 2;
                        auto x_padding_bottom = shape_data[i] - x_size[i] - x_padding_top;

                        padding[i][0] = x_padding_top;
                        padding[i][1] = x_padding_bottom;
                    }
                }
            } else {
                for (int i = 0; i < dims; ++i) {
                    if (shape_data[i] <= 0) {
                        padding[i][0] = -m_shift[i];
                        padding[i][1] = m_shift[i];
                    } else {
                        auto x_padding_top = -m_shift[i];
                        auto x_padding_bottom = m_shift[i] + shape_data[i] - x_size[i];

                        padding[i][0] = x_padding_top;
                        padding[i][1] = x_padding_bottom;
                    }
                }
            }

            auto x_padding_tensor = tensor::build(INT32, {int(dims), 2}, padding[0].data());

            stack.push(0);
            stack.push(x_padding_tensor);

            TS_AUTO_CHECK(1 == RunOperator(m_pad_op, stack, 2));

            return 1;
        }
    }
}

using namespace ts;
using namespace zoo;

TS_REGISTER_OPERATOR(CropND, ts::CPU, ts::name::layer::crop_nd());
