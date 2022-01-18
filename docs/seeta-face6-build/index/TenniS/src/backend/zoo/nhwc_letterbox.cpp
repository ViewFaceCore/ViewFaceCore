//
// Created by kier on 2019-05-28.
//

#include "backend/zoo/nhwc_letterbox.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"
#include "runtime/stack.h"
#include <algorithm>

#include <algorithm>

namespace ts {
    namespace zoo {
        // static const std::string OUTER_VALUE = "outer_value";
        NHWCLetterBox::NHWCLetterBox() {
            field(name::size, REQUIRED);
            field(name::type, OPTIONAL, tensor::from<int32_t>(0));
            field(name::outer_value, OPTIONAL, tensor::from<float>(0));
        }

        void NHWCLetterBox::init() {
            supper::init();

            auto size = tensor::cast(INT32, this->get(name::size));

            TS_AUTO_CHECK(size.has_shape(2) || size.has_shape(1));
            auto size_count = size.count();

            m_size.resize(size_count);
            for (int i = 0; i < size_count; ++i) {
                m_size[i] = size.data<int32_t>()[i];
            }

            auto &context = ctx::ref<DeviceContext>();

            m_sample_op = OperatorCreator::Create(context.computing_device.type(), name::layer::affine_sample2d(), false);

            TS_CHECK_NQ(m_sample_op, nullptr) << "Can not find operator: " << name::layer::affine_sample2d();

            m_sample_op->set(name::type, tensor::clone(INT32, get(name::type)));
            m_sample_op->set(name::outer_value, tensor::clone(FLOAT32, get(name::outer_value)));
            m_sample_op->set(name::dim, tensor::from<int32_t>(1));

            m_sample_op->init();

            m_sample_size = Tensor(INT32, {2, });
            m_sample_affine = Tensor(FLOAT32, {3, 3});

            if (m_size.size() == 2) {
                m_sample_size.data<int32_t>(0) = m_size[1];
                m_sample_size.data<int32_t>(1) = m_size[0];
            } else {
                m_sample_size.data<int32_t>(0) = m_size[0];
                m_sample_size.data<int32_t>(1) = m_size[0];
            }

            float *affine = m_sample_affine.data<float>();
            for (int i = 0; i < 8; ++i) affine[i] = 0;
            affine[8] = 1;
        }

        static inline Size2D infer_size(const Size2D &x, const std::vector<int> &size) {
            if (size.size() == 2) {
                return Size2D(size[1], size[0]);
            } else {
                return Size2D(size[0], size[0]);
            }
        }

        int NHWCLetterBox::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() == 4);

            Size2D x_size(x.size(1), x.size(2));

            Size2D y_size = infer_size(x_size, m_size);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), {-1, y_size.height, y_size.width, -1});

            return 1;
        }

        int NHWCLetterBox::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() == 4);

            Size2D x_size(x.size(1), x.size(2));

            Size2D y_size = infer_size(x_size, m_size);

            auto x_scale = float(y_size.width) / x_size.width;
            auto y_scale = float(y_size.height) / x_size.height;
            auto scale = std::min(x_scale, y_scale);

            m_sample_affine.data<float>(0) = 1 / scale;
            m_sample_affine.data<float>(4) = 1 / scale;

            stack.push(m_sample_size);
            stack.push(m_sample_affine);

            TS_AUTO_CHECK(1 == RunOperator(m_sample_op, stack, 3));

            return 1;
        }
    }
}

using namespace ts;
using namespace zoo;

TS_REGISTER_OPERATOR(NHWCLetterBox, ts::CPU, ts::name::layer::nhwc_letterbox());
