#include <backend/base/base_new_shape.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <global/operator_factory.h>

namespace ts {
    namespace cpu {
        class Squeeze : public OperatorOnAny<base::NewShape> {
        public:
            using self = Squeeze;
            using supper = OperatorOnAny<base::NewShape>;

            Squeeze() {
                field(name::axes, OPTIONAL);
            }

            void init() final {
                supper::init();

                m_axes.clear();
                if (has(name::axes)) {
                    m_axes = tensor::array::to_int(get(name::axes));
                }
            }

            Shape newshape(const Tensor &x) final {
                auto shape = x.sizes();
                if (m_axes.empty()) {
                    auto it = shape.begin();
                    while (it != shape.end()) {
                        if (*it == 1) {
                            it = shape.erase(it);
                        } else {
                            ++it;
                        }
                    }
                } else {
                    for (auto axis_it = m_axes.rbegin(); axis_it != m_axes.rend(); ++axis_it) {
                        auto axis = *axis_it;
                        auto max_axis = int(shape.size());
                        axis = axis >= 0 ? axis : (max_axis + axis);
                        if (axis < 0 || axis >= max_axis) {
                            TS_LOG_ERROR << "Can not squeeze shape " << to_string(x.sizes())
                                         << " with axes=" << to_string(m_axes) << eject;
                        }
                        auto it = shape.begin() + axis;
                        if (*it != 1) {
                            TS_LOG_ERROR << "Can not squeeze shape " << to_string(x.sizes())
                                         << " with axes=" << to_string(m_axes) << eject;
                        }
                        shape.erase(it);
                    }
                }
                return std::move(shape);
            }

        private:
            std::vector<int32_t> m_axes;
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Squeeze, CPU, name::layer::squeeze())
