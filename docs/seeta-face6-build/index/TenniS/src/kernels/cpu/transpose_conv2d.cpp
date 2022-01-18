#include <kernels/cpu/transpose_conv2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Conv2DTranspose, CPU, name::layer::transpose_conv2d())
