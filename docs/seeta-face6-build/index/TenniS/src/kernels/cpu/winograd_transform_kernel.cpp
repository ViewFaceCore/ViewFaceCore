#include <kernels/cpu/winograd_transform_kernel.h>

#include "kernels/cpu/winograd_transform_kernel.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "kernels/cpu/winograd_algorithm.h"

namespace ts {
    namespace cpu {
        void WinogradTransKernel::transform_kernel(const Tensor &x, WinogradConv2DMode winograd_mode, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
            case DTYPE: { \
                if (winograd_mode == F6X6_3X3) \
                    Conv2dWinogradAlgorithm<TYPE>::winograd_f63_transform_and_pack_kernel(x, 64, out); \
                else if(winograd_mode == F2X2_3X3) \
                    Conv2dWinogradAlgorithm<TYPE>::winograd_f23_transform_and_pack_kernel(x, 16, out); \
                break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
//                DECLARE_COMPUTE_RUN(FLOAT64, double);
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
TS_REGISTER_OPERATOR(WinogradTransKernel, ts::CPU, name::layer::winograd_transform_kernel())