#include <kernels/cpu/pooling2d_v2.h>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>

#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"

#include "module/bubble.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace cpu {

        Pooling2DV2::Pooling2DV2() {
            field(name::format, REQUIRED);
            field(name::type, OPTIONAL, tensor::from(int(Pooling2DType::MAX)));
            field(name::padding_type, OPTIONAL, tensor::from(int(Padding2DType::BLACK)));
        }

        void Pooling2DV2::init() {
            supper::init();

            auto &context = ctx::ref<DeviceContext>();

            m_op_pooling2d = OperatorCreator::Create(context.computing_device.type(), name::layer::pooling2d(), false);

            TS_CHECK_NQ(m_op_pooling2d, nullptr) << "Can not find operator: " << name::layer::pooling2d();

            m_op_pooling2d->set(Bubble::RetentionParam::op, tensor::from(name::layer::pooling2d_v2()));
            m_op_pooling2d->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_pooling2d->has(param) && this->has(param)) {
                    m_op_pooling2d->set(param, get(param));
                }
            }

            m_op_pooling2d->set(name::format, get(name::format));
            m_op_pooling2d->set(name::type, get(name::type));
            m_op_pooling2d->set(name::padding_type, get(name::padding_type));
        }

        static bool is_int_equal(const Tensor &lhs, const Tensor &rhs) {
            if (!lhs.has_shape(rhs.sizes())) return false;
            auto count = lhs.count();
            for (int i = 0; i < count; ++i) {
                if(lhs.data<int>(i) != rhs.data<int>(i)) return false;
            }
            return true;
        }

        int Pooling2DV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 4);

            auto padding = tensor::cast(INT32, stack[1]);
            auto ksize = tensor::cast(INT32, stack[2]);
            auto stride = tensor::cast(INT32, stack[3]);

            bool updated = false;
            if (!is_int_equal(padding, m_padding_int4x2)) {
                m_padding_int4x2 = padding.clone();
                m_op_pooling2d->set(name::padding, m_padding_int4x2);
                updated = true;
            }
            if (!is_int_equal(ksize, m_ksize_int4)) {
                m_ksize_int4 = ksize.clone();
                m_op_pooling2d->set(name::ksize, m_ksize_int4);
                updated = true;
            }
            if (!is_int_equal(stride, m_stride_int4)) {
                m_stride_int4 = stride.clone();
                m_op_pooling2d->set(name::stride, m_stride_int4);
                updated = true;
            }

            if (updated) {
                m_op_pooling2d->init();
            }

            stack.push(0);

            return InferOperator(m_op_pooling2d, stack, 1, output);
        }

        int Pooling2DV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 4);

            auto padding = tensor::cast(INT32, stack[1]);
            auto ksize = tensor::cast(INT32, stack[2]);
            auto stride = tensor::cast(INT32, stack[3]);

            bool updated = false;
            if (!is_int_equal(padding, m_padding_int4x2)) {
                m_padding_int4x2 = padding.clone();
                m_op_pooling2d->set(name::padding, m_padding_int4x2);
                updated = true;
            }
            if (!is_int_equal(ksize, m_ksize_int4)) {
                m_ksize_int4 = ksize.clone();
                m_op_pooling2d->set(name::ksize, m_ksize_int4);
                updated = true;
            }
            if (!is_int_equal(stride, m_stride_int4)) {
                m_stride_int4 = stride.clone();
                m_op_pooling2d->set(name::stride, m_stride_int4);
                updated = true;
            }

            if (updated) {
                m_op_pooling2d->init();
            }


            stack.push(0);

            return RunOperator(m_op_pooling2d, stack, 1);
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Pooling2DV2, CPU, name::layer::pooling2d_v2())

TS_REGISTER_FP16_OPERATOR(Pooling2DV2, GPU, name::layer::pooling2d_v2())
