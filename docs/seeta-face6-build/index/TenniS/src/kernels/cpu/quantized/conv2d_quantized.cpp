#include <kernels/cpu/quantized/conv2d_quantized.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Conv2DQuantized, CPU, name::layer::conv2d_quantized())
