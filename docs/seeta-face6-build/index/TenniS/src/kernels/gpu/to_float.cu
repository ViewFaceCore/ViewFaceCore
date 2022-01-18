#include <kernels/gpu/to_float.h>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>

#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"

#include "module/bubble.h"
#include "core/tensor_builder.h"
#include "utils/need.h"



namespace ts {
    namespace gpu {

        ToFloat::ToFloat() {
        }

        void ToFloat::init() {
            supper::init();

            auto &context = ctx::ref<DeviceContext>();

            m_op_castv2 = OperatorCreator::Create(context.computing_device.type(), name::layer::cast(), false);

            TS_CHECK_NQ(m_op_castv2, nullptr) << "Can not find operator: " << name::layer::cast();

            m_op_castv2->set(Bubble::RetentionParam::op, tensor::from(name::layer::conv2d_v2()));
            m_op_castv2->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_castv2->has(param) && this->has(param)) {
                    m_op_castv2->set(param, get(param));
                }
            }

            int dtype = FLOAT32;
            m_op_castv2->set(name::dtype, tensor::from(dtype));

            m_op_castv2->init();
        }


        int ToFloat::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            return m_op_castv2->infer(stack, output);
        }

        int ToFloat::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            return RunOperator(m_op_castv2, stack, 1);
        }


    }
}




using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(ToFloat, GPU, name::layer::to_float())
TS_REGISTER_FP16_OPERATOR(ToFloat, GPU, name::layer::to_float())

