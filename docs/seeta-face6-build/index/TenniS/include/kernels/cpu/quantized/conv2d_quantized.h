#ifndef TENSORSTACK_KERNELS_CPU_QUANTIZED_CONV2D_H
#define TENSORSTACK_KERNELS_CPU_QUANTIZED_CONV2D_H

#include "kernels/cpu/operator_on_cpu.h"
#include "backend/base/base_conv2d_quantized.h"
#include "conv2d_quantized_core.h"


namespace ts {
	namespace cpu {
	    using Conv2DQuantized = base::Conv2DWithCore<OperatorOnCPU<base::Conv2DQuantized>, Conv2DQuantizedCore>;
	}
}


#endif //TENSORSTACK_KERNELS_CPU_QUANTIZED_CONV2D_H