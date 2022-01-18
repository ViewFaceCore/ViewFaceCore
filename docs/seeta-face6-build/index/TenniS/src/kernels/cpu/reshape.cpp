#include <kernels/cpu/reshape.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Reshape, CPU, name::layer::reshape())
