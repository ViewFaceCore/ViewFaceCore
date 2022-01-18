#include <kernels/cpu/shape.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(GetShape, CPU, name::layer::shape())
