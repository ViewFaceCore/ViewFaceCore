#ifndef TENSORSTACK_KERNELS_GPU_RELU_MAX_H
#define TENSORSTACK_KERNELS_GPU_RELU_MAX_H

#include "backend/base/base_relu_max.h"
#include "operator_on_gpu.h"

namespace ts {
	namespace gpu {
		class ReLUMax : public OperatorOnGPU<base::ReLUMax> {
		public:
		    using self = ReLUMax;
			using supper = OperatorOnGPU<base::ReLUMax>;

            void relu_max(const Tensor &x, float max, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_GPU_RELU_MAX_H