#include <kernels/cpu/unsqueeze.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Unsqueeze, CPU, name::layer::unsqueeze())
