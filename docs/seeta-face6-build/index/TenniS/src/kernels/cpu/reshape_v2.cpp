#include <kernels/cpu/reshape_v2.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ReshapeV2, CPU, name::layer::reshape_v2())
