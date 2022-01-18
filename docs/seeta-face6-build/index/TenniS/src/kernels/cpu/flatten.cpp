#include <kernels/cpu/flatten.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <global/operator_factory.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Flatten, CPU, name::layer::flatten())
