#include <kernels/cpu/prelu.h>
#include <core/tensor_builder.h>
#include "backend/name.h"

#include "global/operator_factory.h"

#include <algorithm>

#include "kernels/common/simd.h"
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

namespace ts {
	namespace cpu {

		template <typename T>
		static void cpu_prelu_compute_run(const Tensor &x, const Tensor &slope, int dim, Tensor &out) {
			auto output_shape = out.sizes();
			const T *input_data = x.data<T>();
			T *output_data = out.data<T>();
			const T *slope_data = slope.data<T>();

			int count = out.count();
			// used in CPU
			std::memcpy(output_data, input_data, count * sizeof(T));

			int pre_dims = 1;
			for (int i = 0; i < dim; i++) {
				pre_dims *= output_shape[i];
			}
			int last_dims = 1;
			for (int i = dim + 1; i < output_shape.size(); i++) {
				last_dims *= output_shape[i];
			}

			for (int i = 0; i < pre_dims; i++) {
				for (int j = 0; j < output_shape[dim]; j++) {
					int offset = i * output_shape[dim] * last_dims + j * last_dims;
					T val = slope_data[j];
					for (int k = 0; k < last_dims; k++) {
						output_data[k + offset] = std::max(output_data[k + offset], T(0)) +
													  val * std::min(output_data[k + offset], T(0));
					}
				}
			}
		}

        template <>
        void cpu_prelu_compute_run<float>(const Tensor &x, const Tensor &slope, int dim, Tensor &out) {
            auto output_shape = out.sizes();
            const float *input_data = x.data<float>();
            float *output_data = out.data<float>();
            const float *slope_data = slope.data<float>();

            // used in CPU
            /*std::memcpy(output_data, input_data, count * sizeof(float));*/

            int pre_dims = 1;
            for (int i = 0; i < dim; i++) {
                pre_dims *= output_shape[i];
            }
            int last_dims = 1;
            for (int i = dim + 1; i < output_shape.size(); i++) {
                last_dims *= output_shape[i];
            }

            for (int i = 0; i < pre_dims; i++) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int j = 0; j < output_shape[dim]; j++) {
                    int offset = i * output_shape[dim] * last_dims + j * last_dims;
                    float val = slope_data[j];
                    float32x4 val_x4(val);
                    float32x4 mul_const(float(0.0));
                    //float32x4 mul_const(float(0.5));
                    for (int k = 0; k < last_dims - 3; k += 4) {
                        int index = k + offset;
                        float32x4 input_data_x4(&input_data[index]);
                        float32x4 output_data_x4 = max_float32x4(input_data_x4, mul_const) + val_x4 * min_float32x4(input_data_x4, mul_const);
                        //float32x4 fabs_input_x4(input_data[index], input_data[index+1], input_data[index+2], input_data[index+3]); 
                        //float32x4 output_data_x4 = (fabs_input_x4 + input_data_x4) * mul_const +
                        //    val_x4 * (input_data_x4 - fabs_input_x4) * mul_const;
                        output_data_x4.store(&output_data[index]);
                    }
                    for (int k = last_dims/4*4; k < last_dims; k++)
                    {
                        output_data[k + offset] = std::max(input_data[k + offset], float(0)) +
                            val * std::min(input_data[k + offset], float(0));
                    }
                }
            }
        }

		void PReLU::prelu(const Tensor &x, const Tensor &slope, int dim, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_prelu_compute_run<TYPE>(x, slope, dim, out); break; }
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
TS_REGISTER_OPERATOR(PReLU, CPU, name::layer::prelu())
