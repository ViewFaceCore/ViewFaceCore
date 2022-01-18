#include <kernels/cpu/to_float.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ToFloat, CPU, name::layer::to_float())
