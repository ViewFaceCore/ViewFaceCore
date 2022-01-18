#include <backend/base/base_broadcast_v2.h>
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
        class BroadcastV2 : public OperatorOnCPU<base::BroadcastV2> {
        public:
            using self = BroadcastV2;
            using supper = OperatorOnCPU<base::BroadcastV2>;

            BroadcastV2() = default;

            void broadcast(const Tensor &x, Tensor &out) override;

            void broad_with_bias(const Tensor &x, Tensor &out, int dim) override;

            void broadcast_with_scalar(const Tensor &x, Tensor &out) override;

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

        template <typename T>
        static void memset(T *dst, size_t N, T val) {
            for (size_t i = 0; i < N; ++i) {
                *dst = val;
                ++dst;
            }
        }

        template <typename T1, typename T2>
        static void memset_v2(T1 *dst, size_t N, T1 val) {
            if (N > 1) {
                dst[0] = val;
                dst[1] = val;
                memset<T2>(reinterpret_cast<T2 *>(dst + 2),
                           (N - 2) / 2,
                           *reinterpret_cast<T2 *>(dst));
                if (N % 2) {
                    dst[N - 1] = val;
                }

            } else if (N > 0) {
                dst[0] = val;
            }
        }

        template <>
        void memset(uint32_t *dst, size_t N, uint32_t val) {
            memset_v2<uint32_t, uint64_t>(dst, N, val);
        }

        template <>
        void memset(uint16_t *dst, size_t N, uint16_t val) {
            memset_v2<uint16_t, uint32_t>(dst, N, val);
        }

        template <>
        void memset(uint8_t *dst, size_t N, uint8_t val) {
            memset_v2<uint8_t, uint16_t>(dst, N, val);
        }

        template<typename T>
        static inline void cpu_broadcast_with_scalar(const Tensor &x, Tensor &out) {
            auto val = x.data<T>()[0];
            auto pout = out.data<T>();
            auto count = out.count();

            memset<T>(pout, size_t(count), val);
        }

        template<typename T>
        static inline void cpu_broadcast_with_bias(const Tensor &x, Tensor &out, int dim) {
            auto px = x.data<T>();
            auto pout = out.data<T>();

            auto &out_shape = out.sizes();

            auto number = std::accumulate(out_shape.begin(), out_shape.begin() + dim, 1, std::multiplies<int>());
            auto count = std::accumulate(out_shape.begin() + dim + 1, out_shape.end(), 1, std::multiplies<int>());

            auto channels = out_shape[dim];

            if (count == 1) {
                for (int n = 0; n < number; ++n) {
                    auto pchannels = pout + n * channels;
                    auto pscalar = px;
                    for (int c = 0; c < channels; ++c) {
                        *pchannels = *pscalar;
                        ++pchannels;
                        ++pscalar;
                    }
                }
            } else {
                for (int n = 0; n < number; ++n) {
                    for (int c = 0; c < channels; ++c) {
                        int offset = (n * channels + c) * count;
                        auto local_pout = pout + offset;
                        memset(local_pout, size_t(count), px[c]);
                    }
                }
            }
        }

        void BroadcastV2::broadcast(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { cpu_broadcast_compute_run<TYPE>(x, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void BroadcastV2::broad_with_bias(const Tensor &x, Tensor &out, int dim) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { cpu_broadcast_with_bias<TYPE>(x, out, dim); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void BroadcastV2::broadcast_with_scalar(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { cpu_broadcast_with_scalar<TYPE>(x, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(BroadcastV2, CPU, "broadcast")
