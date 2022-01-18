//
// Created by kier on 2019/3/18.
//

#include <backend/zoo/nhwc_scale_resize2d.h>

#include "backend/zoo/nhwc_scale_resize2d.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"
#include "runtime/stack.h"

namespace ts {
    namespace zoo {
        NHWCScaleResize2D::NHWCScaleResize2D() {
            field(name::size, REQUIRED);
            field(name::type, OPTIONAL, tensor::from<int32_t>(0));
        }

        void NHWCScaleResize2D::init() {
            supper::init();

            auto size = tensor::cast(INT32, this->get(name::size));

            TS_AUTO_CHECK(size.has_shape(2) || size.has_shape(1));
            auto size_count = size.count();

            m_size.resize(size_count);
            for (int i = 0; i < size_count; ++i) {
                m_size[i] = size.data<int32_t>()[i];
            }

            auto &context = ctx::ref<DeviceContext>();

            m_resize2d_op = OperatorCreator::Create(context.computing_device.type(), name::layer::resize2d(), false);

            TS_CHECK_NQ(m_resize2d_op, nullptr) << "Can not find operator: " << name::layer::resize2d();

            m_resize2d_op->set(name::type, tensor::clone(INT32, get(name::type)));

            m_resize2d_op->init();

            m_dynamic_size = tensor::build(INT32, {-1, -1, -1, -1});
        }

        /**
         *
         * @param x
         * @param size {W, H} format
         * @return
         */
        static inline Size2D infer_size(const Size2D &x, const std::vector<int> &size) {
            if (size.size() == 2) {
                return Size2D(size[1], size[0]);
            } else {
                if (x.height > x.width) {
                    return Size2D(size[0] * x.height / x.width, size[0]);
                } else {
                    return Size2D(size[0], size[0] * x.width / x.height);
                }
            }
        }

        int NHWCScaleResize2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() == 4);

            Size2D x_size(x.size(1), x.size(2));

            Size2D y_size = infer_size(x_size, m_size);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), {-1, y_size.height, y_size.width, -1});

            return 1;
        }

        int NHWCScaleResize2D::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];
            TS_AUTO_CHECK(x.dims() == 4);

            Size2D x_size(x.size(1), x.size(2));

            Size2D y_size = infer_size(x_size, m_size);

            m_dynamic_size.data<int32_t>(1) = y_size.height;
            m_dynamic_size.data<int32_t>(2) = y_size.width;

            stack.push(m_dynamic_size);

            TS_AUTO_CHECK(1 == RunOperator(m_resize2d_op, stack, 2));

            return 1;
        }
    }
}

using namespace ts;
using namespace zoo;

TS_REGISTER_OPERATOR(NHWCScaleResize2D, ts::CPU, ts::name::layer::nhwc_scale_resize2d());

