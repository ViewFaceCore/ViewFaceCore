//
// Created by kier on 19-6-17.
//

#include "backend/base/base_activation.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "kernels/cpu/operator_on_cpu.h"
#include "kernels/common/math.h"

namespace ts {
    namespace cpu {
        template<typename T>
        static T sigmoid(T x) {
            return T(T(1) / (T(1) + std::exp(-x)));
        }

        template<typename T>
        static T tanh(T x) {
            return T(2) * sigmoid(T(2) * x) - T(1);
        }

        template<typename T>
        static void cpu_tanh_compute_run(const Tensor &x, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            for (int i = 0; i < count; i++) {
                *output_data = tanh(*input_data);
                output_data++;
                input_data++;
            }
        }

        class Tanh : public OperatorOnCPU<base::Activation> {
        public:
            void active(const Tensor &x, Tensor &out) final {

                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_tanh_compute_run<TYPE>(x, out); break; }
                    // DECLARE_COMPUTE_RUN(INT8, int8_t);
                    // DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                    // DECLARE_COMPUTE_RUN(INT16, int16_t);
                    // DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                    // DECLARE_COMPUTE_RUN(INT32, int32_t);
                    // DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                    // DECLARE_COMPUTE_RUN(INT64, int64_t);
                    // DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                    DECLARE_COMPUTE_RUN(FLOAT32, float);
                    DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                    default: {
                        TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                        break;
                    }
                }
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Tanh, CPU, "tanh")