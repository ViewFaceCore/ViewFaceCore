#include <kernels/cpu/conv2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Conv2D, CPU, name::layer::conv2d())
