#ifndef TENSORSTACK_KERNELS_GPU_DEPTHWISE_CONV2D_H
#define TENSORSTACK_KERNELS_GPU_DEPTHWISE_CONV2D_H

#include "operator_on_gpu.h"
#include "backend/base/base_depthwise_conv2d.h"
#include "depthwise_conv2d_core.h"


namespace ts {
	namespace gpu {
	    using DepthwiseConv2D = base::DepthwiseConv2DWithCore<OperatorOnGPU<base::DepthwiseConv2D>, DepthwiseConv2DCore>;
	}
}


#endif //TENSORSTACK_KERNELS_GPU_DEPTHWISE_CONV2D_H
