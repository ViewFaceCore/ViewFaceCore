#include <kernels/cpu/cast.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(CastV2, CPU, name::layer::cast())
