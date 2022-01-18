#include "backend/base/base_force_color.h"
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
        class ForceColor : public OperatorOnCPU<base::ForceColor> {
        public:
            using self = ForceColor;
            using supper = OperatorOnCPU<base::ForceColor>;

            void force_color(const Tensor &x, Tensor &out) override;
        };
    }
}

namespace ts {
	namespace cpu {
        template<typename T>
        void cpu_force_color_compute_run(const Tensor &x, Tensor &out) {
            auto &size = x.sizes();
            auto dims = x.dims();
            auto number = std::accumulate(size.begin(), size.end() - 1, 1, std::multiplies<int32_t>());
            auto input_channels = x.size(dims - 1);
            auto output_channels = out.size(dims - 1);

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();

            for (int i = 0; i < number; ++i) {
                auto input_pixel = &input_data[i * input_channels];
                auto output_pixel = &output_data[i * output_channels];
                for (int j = 0; j < output_channels; ++j) {
                    output_pixel[j] = input_pixel[0];
                }
            }
        }

        void ForceColor::force_color(const Tensor &x, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_force_color_compute_run<TYPE>(x, out); break; }
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
TS_REGISTER_OPERATOR(ForceColor, ts::CPU, name::layer::force_color())
