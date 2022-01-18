//
// Created by yang on 2019/11/8.
//

#include "kernels/cpu/conv2d_winograd_v2.h"
#include "backend/name.h"
#include "core/tensor.h"
#include "core/tensor_builder.h"
#include "core/device_context.h"
#include "global/operator_factory.h"
#include "module/bubble.h"

namespace ts{
    namespace cpu{

        Conv2DWinogradV2::Conv2DWinogradV2() {
            field(name::format, REQUIRED);
            field(name::padding_value, OPTIONAL, tensor::from(0.0f));
            field(name::kernel_winograd_transformed, OPTIONAL, tensor::from<bool>(false));
        }

        void Conv2DWinogradV2::init() {
            supper::init();

            auto &context = ctx::ref<DeviceContext>();
            m_op_conv2d_winograd = OperatorCreator::Create(context.computing_device.type(), name::layer::conv2d_winograd(), false);
            TS_CHECK_NQ(m_op_conv2d_winograd, nullptr) << "Can not find operator: " << name::layer::conv2d();

            m_op_conv2d_winograd->set(Bubble::RetentionParam::op, tensor::from(name::layer::conv2d_winograd()));
            m_op_conv2d_winograd->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_conv2d_winograd->has(param) && this->has(param)) {
                    m_op_conv2d_winograd->set(param, get(param));
                }
            }

            m_op_conv2d_winograd->set(name::format, get(name::format));
            m_op_conv2d_winograd->set(name::padding_value, get(name::padding_value));
            m_op_conv2d_winograd->set(name::kernel_winograd_transformed, get(name::kernel_winograd_transformed));
        }

        static bool is_int_equal(const Tensor &lhs, const Tensor &rhs) {
            if (!lhs.has_shape(rhs.sizes())) return false;
            auto count = lhs.count();
            for (int i = 0; i < count; ++i) {
                if(lhs.data<int>(i) != rhs.data<int>(i)) return false;
            }
            return true;
        }

        int Conv2DWinogradV2::infer(Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto padding = tensor::cast(INT32, stack[1]);
            if (!is_int_equal(padding, m_int_padding4x2)) {
                m_int_padding4x2 = padding.clone();
                m_op_conv2d_winograd->set(name::padding, m_int_padding4x2);
                m_op_conv2d_winograd->init();
            }

            stack.push(0);
            stack.push(2);

            return InferOperator(m_op_conv2d_winograd, stack, 2, output);
        }

        int Conv2DWinogradV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto padding = tensor::cast(INT32, stack[1]);

            if (!is_int_equal(padding, m_int_padding4x2)) {
                m_int_padding4x2 = padding.clone();
                m_op_conv2d_winograd->set(name::padding, m_int_padding4x2);
                m_op_conv2d_winograd->init();
            }

            stack.push(0);
            stack.push(2);

            return RunOperator(m_op_conv2d_winograd, stack, 2);
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Conv2DWinogradV2, CPU, name::layer::conv2d_winograd_v2())
