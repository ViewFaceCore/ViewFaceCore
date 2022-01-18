#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d.h"
#include "conv2d_core.h"


namespace ts {
	namespace cpu {
	    using Conv2D = base::PackedConv2DWithCore<OperatorOnCPU<base::Conv2D>, Conv2DCore>;
	}
}


#endif //TENSORSTACK_KERNELS_CPU_CONV2D_H