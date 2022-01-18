#include "backend/base/base_force_gray.h"
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
        class ForceGray : public OperatorOnCPU<base::ForceGray> {
        public:
            using self = ForceGray;
            using supper = OperatorOnCPU<base::ForceGray>;

            void force_gray(const Tensor &x, const std::vector<float> &scale, Tensor &out) override;
        };
    }
}

namespace ts {
	namespace cpu {
	    namespace {
	        template <typename T>
	        class __sum_type {
            public:
	            using Type = float;
	        };

	        template <>
	        class __sum_type<double> {
            public:
	            using Type = double;
	        };
	    }

        template<typename T>
        void cpu_force_gray_compute_run(const Tensor &x, const std::vector<float> &scale, Tensor &out) {
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

                typename __sum_type<T>::Type pixel = 0;

                for (int j = 0; j < input_channels; ++j) {
                    pixel += scale[j] * input_pixel[j];
                }

                output_pixel[0] = T(pixel);
            }
        }

        void ForceGray::force_gray(const Tensor &x, const std::vector<float> &scale, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_force_gray_compute_run<TYPE>(x, scale, out); break; }
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
TS_REGISTER_OPERATOR(ForceGray, ts::CPU, name::layer::force_gray())
