#include "backend/base/base_new_shape.h"
#include "core/tensor_builder.h"

#include <numeric>
#include "global/operator_factory.h"

namespace ts {
    namespace cpu {
    class Flatten2D : public OperatorOnAny<base::NewShape> {
        public:
            using self = Flatten2D;
            using supper = NewShape;

            Flatten2D() {
                field("dim", OPTIONAL, tensor::from<int32_t>(1));
            }

            void init() override  {
                m_dim = tensor::to_int(get("dim"));
            }

            Shape newshape(const Tensor &x) final {
                auto fixed_dim = m_dim;
                if (fixed_dim < 0) fixed_dim += x.dims();
                auto size = x.sizes();

                if (fixed_dim <= 0) {
                    return {1, std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>())};
                } else if (fixed_dim >= x.dims()) {
                    return {std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>()), 1};
                } else {
                    return {std::accumulate(size.begin(), size.begin() + fixed_dim, 1, std::multiplies<int>()),
                               std::accumulate(size.begin() +  fixed_dim, size.end(), 1, std::multiplies<int>())};
                }
            }

        private:
            int m_dim = 1;
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Flatten2D, CPU, "flatten2d")
