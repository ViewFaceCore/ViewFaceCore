#include "backend/base/base_norm_image.h"
#include "kernels/cpu/operator_on_cpu.h"

#include <core/tensor_builder.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>
#include <math.h>

#include <kernels/common/simd.h>

namespace ts {
    namespace cpu {
        class NormImage : public OperatorOnCPU<base::NormImage> {
        public:
            using self = NormImage;
            using supper = OperatorOnCPU<base::NormImage>;

            void norm_image(const Tensor &x, float epsilon, Tensor &out) override;
        };
    }
}

namespace ts {
	namespace cpu {
        template<typename T>
        void cpu_norm_image_compute_run(const Tensor &x, float epsilon, Tensor &out) {
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
                // std_dev = std::max<T>(std_dev, 1 / std::sqrt(count));
				double std_dev_rec = 1 / (std_dev + epsilon);

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

		void NormImage::norm_image(const Tensor &x, float epsilon, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_norm_image_compute_run<TYPE>(x, epsilon, out); break; }
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
TS_REGISTER_OPERATOR(NormImage, ts::CPU, name::layer::norm_image())
