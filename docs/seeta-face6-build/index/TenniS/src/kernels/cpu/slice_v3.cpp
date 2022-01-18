//
// Created by kier on 2019/4/8.
//

#include "backend/base/base_slice_v3.h"
#include "kernels/cpu/operator_on_cpu.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include "core/tensor_iterator.h"

#include <numeric>

namespace ts {
    namespace cpu {
        class SliceV3 : public OperatorOnCPU<base::SliceV3> {
        public:
            using self = SliceV3;
            using supper = OperatorOnCPU<base::SliceV3>;

            void slice(
                    const Tensor &x,
                    const std::vector<int> &begin,
                    const std::vector<int> &end,
                    const std::vector<int> &stride,
                    Tensor &out) override;
        };

        static void insert_back_zeros(std::vector<int> &arr, size_t count) {
            std::vector<int> zeros(count, 0);
            arr.insert(arr.end(), zeros.begin(), zeros.end());
        }

        static void insert_back_ones(std::vector<int> &arr, size_t count) {
            std::vector<int> ones(count, 1);
            arr.insert(arr.end(), ones.begin(), ones.end());
        }

        template<typename T>
        static void cpu_compute_strided_slice(const ts::Tensor &x, const std::vector<int> &begin,
                                              const std::vector<int> &end, const std::vector<int> &stride,
                                              ts::Tensor &out) {
            auto &x_shape = x.sizes();

            auto fixed_begin = begin;
            insert_back_zeros(fixed_begin, x_shape.size() - begin.size());
            auto fixed_stride = stride;
            insert_back_ones(fixed_stride, x_shape.size() - stride.size());

            auto in_data = x.data<T>();
            auto out_data = out.data<T>();

            HypeShape hype_shape(x.sizes());
            ShapeIterator shape_it(out.sizes());
            auto shape_count = out.count();
            std::vector<int> coord_in(out.dims());
            for (int i = 0; i < shape_count; ++i) {
                auto &coord_out = shape_it.coordinate();
                for (size_t j = 0; j < coord_out.size(); ++j) {
                    coord_in[j] = coord_out[j] * fixed_stride[j] + fixed_begin[j];
                }
                auto out_index = i;
                auto in_index = hype_shape.to_index(coord_in);

                out_data[out_index] = in_data[in_index];

                ++shape_it;
            }
        }

        void SliceV3::slice(const ts::Tensor &x, const std::vector<int> &begin,
                            const std::vector<int> &end, const std::vector<int> &stride,
                            ts::Tensor &out) {
            class uint128_t {
            public:
                uint64_t h;
                uint64_t l;
            };
            auto type_bytes = out.proto().type_bytes();
            switch (type_bytes) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_compute_strided_slice<TYPE>(x, begin, end, stride, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
                DECLARE_COMPUTE_RUN(16, uint128_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    auto dtype = out.dtype();
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
TS_REGISTER_OPERATOR(SliceV3, CPU, "slice_v3")
