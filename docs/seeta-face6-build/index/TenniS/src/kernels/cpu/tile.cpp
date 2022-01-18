//
// Created by kier on 19-7-23.
//

#include "backend/base/base_tile.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "kernels/cpu/operator_on_cpu.h"
#include "kernels/common/math.h"

#include "backend/name.h"

#include "core/tensor_iterator.h"

#include <numeric>

namespace ts {
    namespace cpu {
        template<typename T>
        static void cpu_tile_compute_run(const Tensor &x, const std::vector<int32_t> &repeats, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();

            auto &x_shape = x.sizes();

            ShapeIterator x_iter(x.sizes());
            ShapeIterator r_iter(repeats);
            HypeShape o_hype(out.sizes());

            std::vector<int32_t> o_index(out.dims());

            auto r_count = std::accumulate(repeats.begin(), repeats.end(), 1, std::multiplies<int32_t>());

            int count = x.count();
            for (int i = 0; i < count; ++i) {
                auto &x_index = x_iter.coordinate();
                for (int j = 0; j < r_count; ++j) {
                    auto &r_index = r_iter.coordinate();
                    for (size_t k = 0; k < o_index.size(); ++k) {
                        o_index[k] = r_index[k] * x_shape[k] + x_index[k];
                    }
                    // set output22
                    output_data[o_hype.to_index(o_index)] = input_data[i];

                    ++r_iter;
                }
                ++x_iter;
            }
        }

        class Tile : public OperatorOnCPU<base::Tile> {
        public:
            void tile(const Tensor &x, const std::vector<int32_t> &repeats, Tensor &out) final {

                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_tile_compute_run<TYPE>(x, repeats, out); break; }
                    DECLARE_COMPUTE_RUN(INT8, int8_t);
                    DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                    DECLARE_COMPUTE_RUN(INT16, int16_t);
                    DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                    DECLARE_COMPUTE_RUN(INT32, int32_t);
                    DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                    DECLARE_COMPUTE_RUN(INT64, int64_t);
                    DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                    DECLARE_COMPUTE_RUN(FLOAT32, float);
                    DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                    default: {
                        TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                        break;
                    }
                }
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Tile, CPU, name::layer::tile())