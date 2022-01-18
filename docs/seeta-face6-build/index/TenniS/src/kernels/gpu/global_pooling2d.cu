#include <kernels/cpu/global_pooling2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>


#include "kernels/gpu/operator_on_gpu.h"
#include "backend/base/base_global_pooling2d.h"
#include "kernels/gpu/pooling2d_core.h"


namespace ts {
    namespace gpu {
        using GlobalPooling2D = base::Pooling2DWithCore<OperatorOnGPU<base::GlobalPooling2D>, Pooling2DCore>;
    }
}


using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(GlobalPooling2D, GPU, name::layer::global_pooling2d())

