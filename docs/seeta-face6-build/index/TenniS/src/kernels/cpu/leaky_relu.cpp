#include "backend/base/base_leaky_relu.h"
#include "kernels/cpu/operator_on_cpu.h"
#include <algorithm>

#include "backend/name.h"
#include "global/operator_factory.h"

#include "kernels/common/simd.h"
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

namespace ts {
	namespace cpu {
        class LeakyReLU : public OperatorOnCPU<base::LeakyReLU> {
        public:
            using self = LeakyReLU;
            using supper = OperatorOnCPU<base::LeakyReLU>;

            void leaky_relu(const Tensor &x, float scale, Tensor &out) override;
        };

		template<typename T>
		static void cpu_leay_relu_compute_run(const Tensor &x, float scale, Tensor &out) {
			const T *input_data = x.data<T>();
			T *output_data = out.data<T>();
			int count = out.count();

            T casted_scale = T(scale);
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
			for (int i = 0; i < count; i++) {
                T val = input_data[i];
                output_data[i] = val > 0 ? val : val * casted_scale;
            }
		}

        template<>
        void cpu_leay_relu_compute_run<float>(const Tensor &x, float scale, Tensor &out) {
            const float *input_data = x.data<float>();
            float *output_data = out.data<float>();
            int count = out.count();
            int count_4 = count / 4;
            //std::memcpy(output_data, input_data, count * sizeof(float));

            float casted_scale = float(scale);
            float32x4 casted_scale_x4(casted_scale);
            int counts = out.count();
            float32x4 const_num_x4(float(0.0));
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int i = 0; i < count_4; ++i) {
                auto input_at = input_data + i * 4;
                auto output_at = output_data + i * 4;

                float32x4 input_data_x4(input_at);
                float32x4 output_data_x4 = max_float32x4(input_data_x4, const_num_x4) + casted_scale_x4 * min_float32x4(input_data_x4, const_num_x4);

                output_data_x4.store(output_at);
            }
            for (int i = count_4 * 4; i < counts; i++) {
                float val = input_data[i];
                output_data[i] = val > 0 ? val : val * casted_scale;
            }
        }

		void LeakyReLU::leaky_relu(const Tensor &x, float scale, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_leay_relu_compute_run<TYPE>(x, scale, out); break; }
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
TS_REGISTER_OPERATOR(LeakyReLU, ts::CPU, "leaky_relu")
