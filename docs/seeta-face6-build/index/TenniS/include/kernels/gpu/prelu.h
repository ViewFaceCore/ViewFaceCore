#ifndef TENSORSTACK_KERNELS_GPU_PRELU_H
#define TENSORSTACK_KERNELS_GPU_PRELU_H

#include "backend/base/base_prelu.h"
#include "operator_on_gpu.h"

namespace ts {
	namespace gpu {
		class PReLU : public OperatorOnGPU<base::PReLU> {
		public:
		    using self = PReLU;
			using supper = OperatorOnGPU<base::PReLU>;

            void prelu(const Tensor &x, const Tensor &slope, int dim, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_GPU_PRELU_H