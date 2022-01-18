#ifndef TENSORSTACK_KERNELS_GPU_EXP_H
#define TENSORSTACK_KERNELS_GPU_EXP_H

#include "backend/base/base_activation.h"
#include "operator_on_gpu.h"

namespace ts {
	namespace gpu {
		class Exp : public OperatorOnGPU<base::Activation> {
		public:
		    using self = Exp;
			using supper = OperatorOnGPU<base::Activation>;

            void active(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_GPU_EXP_H
