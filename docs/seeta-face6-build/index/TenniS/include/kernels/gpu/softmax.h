#ifndef TENSORSTACK_KERNELS_GPU_SOFTMAX_H
#define TENSORSTACK_KERNELS_GPU_SOFTMAX_H

#include "backend/base/base_softmax.h"
#include "operator_on_gpu.h"

namespace ts {
	namespace gpu {
		class Softmax : public OperatorOnGPU<base::Softmax> {
		public:
			using self = Softmax;
			using supper = OperatorOnGPU<base::Softmax>;

			void softmax(const Tensor &x, int dim, bool smooth, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_GPU_SOFTMAX_H