#ifndef TENSORSTACK_KERNELS_GPU_SIGMOID_H
#define TENSORSTACK_KERNELS_GPU_SIGMOID_H

#include "backend/base/base_sigmoid.h"
#include "operator_on_gpu.h"

namespace ts {
	namespace gpu {
		class Sigmoid : public OperatorOnGPU<base::Sigmoid> {
		public:
		    using self = Sigmoid;
			using supper = OperatorOnGPU<base::Sigmoid>;

            void active(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_GPU_SIGMOID_H