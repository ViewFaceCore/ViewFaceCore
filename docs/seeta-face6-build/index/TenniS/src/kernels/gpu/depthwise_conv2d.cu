#include <kernels/gpu/depthwise_conv2d.h>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(DepthwiseConv2D, GPU, name::layer::depthwise_conv2d())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(DepthwiseConv2D, ts::GPU, name::layer::depthwise_conv2d())
#endif
