#include <backend/base/base_broadcast.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include <core/memory.h>
#include <numeric>
#include <kernels/cpu/operator_on_cpu.h>

#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif


namespace ts {
    namespace cpu {
        class Broadcast : public OperatorOnCPU<base::Broadcast> {
        public:
            using self = Broadcast;
            using supper = OperatorOnCPU<base::Broadcast>;

            Broadcast() = default;

            void broadcast(const Tensor &x, const std::vector<int32_t> &shape, Tensor &out) override;
        };

        static inline int to_mod_index(const HypeShape &hype, const Shape &coordinate) {
            auto temp = coordinate;
            for (size_t i = 0; i < temp.size(); ++i) {
                temp[i] %= hype.shape(i);
            }
            return hype.to_index(temp);
        }

        template<typename T>
        static inline void cpu_broadcast_compute_run(const Tensor &C, Tensor &out) {
            HypeShape C_hype(C.sizes());
            ShapeIterator out_iterator(out.sizes());

            auto pC = C.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();
            for (int i = 0; i < ncount; i++) {
                auto &tmpshape = out_iterator.coordinate();
                pout[i] = pC[to_mod_index(C_hype, tmpshape)];
                ++out_iterator;
            }
        }

        void Broadcast::broadcast(const Tensor &x, const std::vector<int32_t> &shape, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_broadcast_compute_run<TYPE>(x, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, int8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, int16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, int32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, int64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, int32_t);
                DECLARE_COMPUTE_RUN(FLOAT64, int64_t);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
// TS_REGISTER_OPERATOR(Broadcast, CPU, "broadcast")
