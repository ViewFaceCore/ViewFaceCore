#include "backend/base/base_reduce_sum.h"
#include "kernels/cpu/operator_on_cpu.h"

#include <core/tensor_builder.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>
#include <math.h>
#include <numeric>

#include <kernels/common/simd.h>

namespace ts {
    namespace cpu {
        class ReduceSum : public OperatorOnCPU<base::ReduceSum> {
        public:
            using self = ReduceSum;
            using supper = OperatorOnCPU<base::ReduceSum>;

            void reduce(const Tensor &x, int dim, Tensor &out) override;
        };
    }
}

namespace ts {
	namespace cpu {
        template<typename T>
        void cpu_reduce_sum_compute_run(const Tensor &x, int dim, Tensor &out) {
            auto &size = x.sizes();
            auto number = std::accumulate(size.begin(), size.begin() + dim, 1, std::multiplies<int32_t>());
            auto channels = size[dim];
            auto width = std::accumulate(size.begin() + dim + 1, size.end(), 1, std::multiplies<int32_t>());

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();
            auto output_count = out.count();

            std::memset(output_data, 0, output_count * out.proto().type_bytes());

            for (int i = 0; i < number; ++i) {
                for (int c = 0; c < channels; ++c) {
                    auto local_input_data = input_data + (i * channels + c) * width;
                    auto local_output_data = output_data + i * width;
                    for (int w = 0; w < width; ++w) {
                        *local_output_data += *local_input_data;
                        ++local_input_data;
                        ++local_output_data;
                    }
                }
            }
        }

        void ReduceSum::reduce(const Tensor &x, int dim, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_reduce_sum_compute_run<TYPE>(x, dim, out); break; }
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
	}
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ReduceSum, ts::CPU, name::layer::reduce_sum())
