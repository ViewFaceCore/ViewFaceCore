#ifndef TENSORSTACK_KERNELS_GPU_SQUARE_H
#define TENSORSTACK_KERNELS_GPU_SQUARE_H

#include "backend/base/base_activation.h"
#include "operator_on_gpu.h"

namespace ts {
	namespace gpu {
		class Square : public OperatorOnGPU<base::Activation> {
		public:
		    using self = Square;
   	            using supper = OperatorOnGPU<base::Activation>;

            void active(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_GPU_SQUARE_H
