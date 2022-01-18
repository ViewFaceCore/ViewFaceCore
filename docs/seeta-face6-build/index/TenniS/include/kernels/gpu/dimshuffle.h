#ifndef TENSORSTACK_KERNELS_GPU_DIMSHUFFLE_H
#define TENSORSTACK_KERNELS_GPU_DIMSHUFFLE_H

#include "operator_on_gpu.h"
#include "backend/base/base_dimshuffle.h"


namespace ts {
	namespace gpu {
		class Dimshuffle : public OperatorOnGPU<base::Dimshuffle> {
		public:
			using self = Dimshuffle;
			using supper = OperatorOnGPU<base::Dimshuffle>;

            Dimshuffle() = default;

            void dimshuffle(const Tensor &x, int dim, const std::vector<int> &shuffle, Tensor &out) override;
		};
	}
}


#endif //TENSORSTACK_KERNELS_GPU_DIMSHUFFLE_H