#include <kernels/cpu/sigmoid.h>
#include <algorithm>

#include "backend/name.h"
#include "global/operator_factory.h"

#include "kernels/common/simd.h"
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

namespace ts {
    namespace cpu {
		template <typename T>
		static T neg(T a) { return -a; }

		template <> uint8_t neg(uint8_t a) { return a; }
		template <> uint16_t neg(uint16_t a) { return a; }
		template <> uint32_t neg(uint32_t a) { return a; }
		template <> uint64_t neg(uint64_t a) { return a; }

        template<typename T>
        static void cpu_sigmoid_compute_run(const Tensor &x, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            //std::memcpy(output_data, input_data, count * sizeof(T));
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int i = 0; i < count; i++) {

                output_data[i] = T(1. / (1. + exp(neg(input_data[i]))));
            }
        }

//#ifdef TS_USE_SSE
//        template<>
//        static void cpu_sigmoid_compute_run<float>(const Tensor &x, Tensor &out) {
//            const float *input_data = x.data<float>();
//            float *output_data = out.data<float>();
//            int count = out.count();
//            //std::memcpy(output_data, input_data, count * sizeof(float));
//            float32x4 const_one(float(1.0));
//            for (int i = 0; i < count - 3; i += 4) {
//                float32x4 val_x4(input_data);
//                float32x4 exp_val_x4(exp(-(input_data[i])), exp(-(input_data[i+1])), exp(-(input_data[i+2])), exp(-(input_data[i+3])));
//                float32x4 output_x4 = const_one / (const_one + exp_val_x4);
//                output_x4.store(output_data);
//                output_data += 4;
//            }
//            for (int i = count/4*4; i < count; i++)
//            {
//                float val = *input_data++;
//                *output_data = 1. / (1. + exp(-(val)));
//                output_data++;
//            }
//        }
//#endif

        void Sigmoid::active(const Tensor &x, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_sigmoid_compute_run<TYPE>(x, out); break; }
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
TS_REGISTER_OPERATOR(Sigmoid, ts::CPU, name::layer::sigmoid())
