#include <kernels/cpu/prewhiten.h>
#include <algorithm>
#include "global/operator_factory.h"

#include "backend/name.h"

namespace ts {
	namespace cpu {

		template<typename T>
		static void cpu_pre_whiten_compute_run(const Tensor &x, Tensor &out) {
			// auto output_shape = out.sizes();
			const T *input_data = x.data<T>();
			T *output_data = out.data<T>();
			auto count = size_t(out.count());
			std::memcpy(output_data, input_data, count * sizeof(T));

			T *at = nullptr;

			// fot batch
			auto batch = x.size(0);
			count /= batch;
			auto batch_outout_data = output_data;

			for (int n = 0; n < batch; ++n) {
				double mean = 0;
				double std_dev = 0;
				
                at = batch_outout_data;
				for (size_t i = 0; i < count; ++i, ++at) mean += *at;
				mean /= count;

				at = batch_outout_data;
				for (size_t i = 0; i < count; ++i, ++at) std_dev += (*at - mean) * (*at - mean);
				std_dev = std::sqrt(std_dev / count);
				std_dev = std::max<double>(std_dev, 1 / std::sqrt(count));
				double std_dev_rec = 1 / std_dev;

				auto t_mean = T(mean);
				auto t_std_dev_rec = T(std_dev_rec);

				at = batch_outout_data;
				for (size_t i = 0; i < count; ++i, ++at) {
					*at -= t_mean;
					*at *= t_std_dev_rec;
				}

				batch_outout_data += count;
			}
		}

		void PreWhiten::prewhiten(const Tensor &x, Tensor &out) {
			auto dtype = out.dtype();
			switch (dtype) {
#define DECLARE_TYPE_AND_RUN(DTYPE, TYPE) \
				case DTYPE: { cpu_pre_whiten_compute_run<TYPE>(x, out); break; }
                // DECLARE_TYPE_AND_RUN(INT8, int8_t);
                // DECLARE_TYPE_AND_RUN(UINT8, uint8_t);
                // DECLARE_TYPE_AND_RUN(INT16, int16_t);
                // DECLARE_TYPE_AND_RUN(UINT16, uint16_t);
                // DECLARE_TYPE_AND_RUN(INT32, int32_t);
                // DECLARE_TYPE_AND_RUN(UINT32, uint32_t);
                // DECLARE_TYPE_AND_RUN(INT64, int64_t);
                // DECLARE_TYPE_AND_RUN(UINT64, uint64_t);
				DECLARE_TYPE_AND_RUN(FLOAT32, float);
				DECLARE_TYPE_AND_RUN(FLOAT64, double);
#undef DECLARE_TYPE_AND_RUN
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
TS_REGISTER_OPERATOR(PreWhiten, CPU, name::layer::prewhiten())
