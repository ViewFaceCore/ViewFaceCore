#include <kernels/cpu/global_pooling2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(GlobalPooling2D, CPU, name::layer::global_pooling2d())

