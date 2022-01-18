#include <kernels/cpu/range.h>
#include <global/operator_factory.h>
#include <backend/name.h>

#include "core/tensor_builder.h"

namespace ts {
    namespace cpu {

        Range::Range() {
        }

        void Range::init() {
            supper::init();
        }

        template<typename T>
        static int cpu_range_infer(const Tensor &start_tensor, const Tensor & limit_tensor, const Tensor & delta_tensor) {
            int steps = 0;
            T start = start_tensor.data<T>()[0];
            T limit = limit_tensor.data<T>()[0];
            T delta = delta_tensor.data<T>()[0];

            steps = int((limit - start) / delta);
           
            if(steps * delta + start  < limit)  {
                steps++;
            }
            return steps;
        }

        template<typename T>
        static void cpu_range_compute_run(const Tensor &start_tensor, const Tensor & limit_tensor, const Tensor & delta_tensor, Tensor &out_tensor) {
            T * pout = out_tensor.data<T>();

            T start = start_tensor.data<T>()[0];
            // T limit = limit_tensor.data<T>()[0];
            T delta = delta_tensor.data<T>()[0];

            for(int i = 0; i<out_tensor.count(); i++) {
                pout[i] = start + i * delta;
            }
        }

        int Range::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 3);
            auto start = stack[0];
            auto limit = stack[1];
            auto delta = stack[2];

            TS_AUTO_CHECK(start.count() == 1);
            TS_AUTO_CHECK(limit.count() == 1);
            TS_AUTO_CHECK(delta.count() == 1);

            int steps = 0; 
            DTYPE dtype = start.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { steps = cpu_range_infer<TYPE>(start, limit, delta); break; }
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

            Shape shape;
            shape.resize(1);
            shape[0] = steps;

            output.resize(1);
            output[0] = Tensor::Prototype(start.dtype(), shape);
            
            return 1;
        }

        int Range::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);
            std::vector<Tensor::Prototype> output;
            infer(stack, output); 

            auto memory_device = running_memory_device();
            auto start = stack[0];
            auto limit = stack[1];
            auto delta = stack[2];

            TS_AUTO_CHECK(start.count() == 1);
            TS_AUTO_CHECK(limit.count() == 1);
            TS_AUTO_CHECK(delta.count() == 1);

            auto out = *stack.push(output[0], memory_device);
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_range_compute_run<TYPE>(start, limit, delta, out); break; }
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


            return 1;
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Range, CPU, name::layer::range())
