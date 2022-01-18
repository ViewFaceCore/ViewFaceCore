#include "backend/base/operator_on_device.h"
#include <global/operator_factory.h>
#include <backend/name.h>

#include <backend/common_structure.h>
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"

#include "module/bubble.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace cpu {
        class Sample2DV2 : public OperatorOnAny<Operator> {
        public:
            using self = Sample2DV2;
            using supper = OperatorOnAny<Operator>;

            Sample2DV2();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Operator::shared m_op_resize2d;
        };

        Sample2DV2::Sample2DV2() {
            field(name::type, OPTIONAL, tensor::from<int32_t>(int32_t(Resize2DType::HARD)));
        }

        void Sample2DV2::init() {
            supper::init();

            auto &context = ctx::ref<DeviceContext>();

            m_op_resize2d = OperatorCreator::Create(context.computing_device.type(), name::layer::resize2d(), false);

            TS_CHECK_NQ(m_op_resize2d, nullptr) << "Can not find operator: " << name::layer::resize2d();

            m_op_resize2d->set(Bubble::RetentionParam::op, tensor::from(name::layer::resize_nearest_neighbor()));
            m_op_resize2d->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_resize2d->has(param) && this->has(param)) {
                    m_op_resize2d->set(param, get(param));
                }
            }

            auto type = get(name::type);
            m_op_resize2d->set(name::type, type);

            m_op_resize2d->init();
        }

        static Tensor GetSizeTensor(Operator *op, Stack &stack, const Tensor &x, const Tensor &scale) {
            auto shape = x.sizes();
            auto scale_count = scale.count();
            if (shape.size() != size_t(scale_count)) {
                TS_LOG_ERROR << op->op() << ":" << op->name() << " scale must has same shape with input tensor, got input: "
                    << x.proto() << ", " << scale.proto() << eject;
            }
            auto size = stack.make(INT32, {scale_count});
            auto float_scale = tensor::cast(FLOAT32, scale);
            for (int32_t i = 0; i < scale_count; ++i) {
                size.data<int32_t>(i) = int32_t(std::floor(shape[i] * float_scale.data<float>(i)));
            }
            return size;
        }


        int Sample2DV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            stack.push(0);
            stack.push(GetSizeTensor(this, stack, stack[0], stack[1]));

            return InferOperator(m_op_resize2d, stack, 2, output);
        }

        int Sample2DV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            stack.push(0);
            stack.push(GetSizeTensor(this, stack, stack[0], stack[1]));

            return RunOperator(m_op_resize2d, stack, 2);
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Sample2DV2, CPU, "sample2d_v2")
