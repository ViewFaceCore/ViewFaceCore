#include "backend/base/base_l2_norm.h"
#include "kernels/cpu/operator_on_cpu.h"

#include <core/tensor_builder.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>
#include <math.h>

#include <kernels/common/simd.h>

namespace ts {
    namespace cpu {
        class L2Norm : public OperatorOnCPU<base::L2Norm> {
        public:
            using self = L2Norm;
            using supper = OperatorOnCPU<base::L2Norm>;

            void normalize(const Tensor &x, int dim, float epsilon, Tensor &out) override;
        };
    }
}

namespace ts {
	namespace cpu {
        template<typename T>
        void cpu_l2_normalize_compute_run(const Tensor &x, int m_dim, float epsilon, Tensor &out) {
            auto &output_shape = out.sizes();

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();

            int body_num = output_shape[m_dim];

            if (body_num == 1) {
                T one(1);
                memset(output_data, out.device(), out.count() * out.proto().type_bytes(),
                       &one, Device(CPU), sizeof(T));
                return;
            }

            int head_num = 1;
            for (int i = 0; i < m_dim; i++) {
                head_num *= output_shape[i];
            }
            int tail_num = 1;
            for (int i = m_dim + 1; i < output_shape.size(); i++) {
                tail_num *= output_shape[i];
            }


            HypeShape hype({head_num, body_num, tail_num});

            auto this_epsilon = T(epsilon);

            // as NCW format
            for (int n = 0; n < head_num; ++n) {
                for (int w = 0; w < tail_num; ++w) {
                    auto channel_index = hype.to_index(n, 0, w);
                    const T *input_channel_data = &input_data[channel_index];
                    T *output_channel_data = &output_data[channel_index];
                    const T *loop_in = input_channel_data;
                    T *loop_out = output_channel_data;
                    T sum = 0;
                    for (int i = 0; i < body_num; ++i) {
                        T data = T(*loop_in * *loop_in);
                        sum += data;
                        // *loop_out = *loop_in;

                        loop_in += tail_num;
                        loop_out += tail_num;
                    }
                    T norm = T(std::sqrt(sum + this_epsilon));
                    loop_in = input_channel_data;
                    loop_out = output_channel_data;
                    for (int i = 0; i < body_num; ++i) {
                        *loop_out = *loop_in / norm;

                        loop_in += tail_num;
                        loop_out += tail_num;
                    }
                }
            }
        }

		template<typename T>
		void cpu_l2_norm_compute_run(const Tensor &x, int m_dim, float epsilon, Tensor &out) {
		    return cpu_l2_normalize_compute_run<T>(x, m_dim, epsilon, out);
		}

		void L2Norm::normalize(const Tensor &x, int dim, float epsilon, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_l2_norm_compute_run<TYPE>(x, dim, epsilon, out); break; }
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
TS_REGISTER_OPERATOR(L2Norm, ts::CPU, name::layer::l2_norm())
