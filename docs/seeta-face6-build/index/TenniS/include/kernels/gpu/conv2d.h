#ifndef TENSORSTACK_KERNELS_GPU_CONV2D_H
#define TENSORSTACK_KERNELS_GPU_CONV2D_H

#include "operator_on_gpu.h"
#include "backend/base/base_conv2d.h"
#include "conv2d_core.h"


namespace ts {
	namespace gpu {
	    using Conv2D = base::Conv2DWithCore<OperatorOnGPU<base::Conv2D>, Conv2DCore>;
	}
}


#endif //TENSORSTACK_KERNELS_GPU_CONV2D_H
