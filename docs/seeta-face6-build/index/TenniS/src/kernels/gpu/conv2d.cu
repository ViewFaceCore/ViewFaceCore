#include <kernels/gpu/conv2d.h>
#include <global/operator_factory.h>
#include "global/fp16_operator_factory.h"
#include <backend/name.h>

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Conv2D, GPU, name::layer::conv2d())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Conv2D, ts::GPU, name::layer::conv2d())
#endif
