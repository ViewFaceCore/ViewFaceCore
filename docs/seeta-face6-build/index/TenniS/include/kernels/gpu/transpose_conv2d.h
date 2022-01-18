#ifndef TENSORSTACK_KERNELS_GPU_CONV2DTranspose_H
#define TENSORSTACK_KERNELS_GPU_CONV2DTranspose_H

#include "operator_on_gpu.h"
#include "backend/base/base_conv2d_transpose.h"
#include "transpose_conv2d_core.h"


namespace ts {
	namespace gpu {
	    using Conv2DTranspose = base::Conv2DTransposeWithCore<OperatorOnGPU<base::Conv2DTranspose>, Conv2DTransposeCore>;
	}
}


#endif
