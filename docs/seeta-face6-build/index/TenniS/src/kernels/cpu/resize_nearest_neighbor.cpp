#include <kernels/cpu/resize_nearest_neighbor.h>
#include <global/operator_factory.h>
#include <backend/name.h>

#include <backend/common_structure.h>
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"

#include "module/bubble.h"
#include "core/tensor_builder.h"
#include "utils/need.h"

namespace ts {
    namespace cpu {

        Resize_Nearest_Neighbor::Resize_Nearest_Neighbor() {
            field(name::align_corners, OPTIONAL);
            field(name::dim, REQUIRED);
            m_align_corners = 0;
        }

        void Resize_Nearest_Neighbor::init() {
            supper::init();

            if(has(name::align_corners)) {
                m_align_corners = tensor::to_int(get(name::align_corners));
            }

            m_dim = tensor::to_int(get(name::dim));

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

            int type = (int)Resize2DType::NEAREST;
            m_op_resize2d->set(name::type, tensor::from(type));
        }


        int Resize_Nearest_Neighbor::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);
            auto size_tensor = tensor::cast(INT32, stack[1]);
            TS_AUTO_CHECK(size_tensor.count() == 2);

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);
            Shape x_shape = x.sizes();

            TS_AUTO_CHECK(m_dim >= 0);
            TS_AUTO_CHECK(m_dim <= x_shape.size() - 2);
            Shape tmpshape(x_shape.size(), -1);

            tmpshape[m_dim] = size_tensor.data<int32_t>()[0];
            tmpshape[m_dim+1] = size_tensor.data<int32_t>()[1];

            TS_AUTO_CHECK((tmpshape[m_dim] > 0) && (tmpshape[m_dim + 1] > 0));

            Tensor newsize_tensor = tensor::from<int32_t>(tmpshape);

            m_op_resize2d->init();
            stack.push(0);
            stack.push(newsize_tensor);

            return InferOperator(m_op_resize2d, stack, 2, output);
        }

        int Resize_Nearest_Neighbor::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto size_tensor = tensor::cast(INT32, stack[1]);
            TS_AUTO_CHECK(size_tensor.count() == 2);
            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);
            Shape x_shape = x.sizes();

            TS_AUTO_CHECK(m_dim >= 0);
            TS_AUTO_CHECK(m_dim <= x_shape.size() - 2);

            Shape tmpshape(x_shape.size(), -1);

            tmpshape[m_dim] = size_tensor.data<int32_t>()[0];
            tmpshape[m_dim+1] = size_tensor.data<int32_t>()[1];

            TS_AUTO_CHECK((tmpshape[m_dim] > 0) && (tmpshape[m_dim + 1] > 0));

            Tensor newsize_tensor = tensor::from<int32_t>(tmpshape);

            m_op_resize2d->init();

            stack.push(0);
            stack.push(newsize_tensor);

            return RunOperator(m_op_resize2d, stack, 2);
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Resize_Nearest_Neighbor, CPU, name::layer::resize_nearest_neighbor())
