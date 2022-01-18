#ifndef TENSORSTACK_KERNELS_CPU_RELU_MAX_H
#define TENSORSTACK_KERNELS_CPU_RELU_MAX_H

#include "backend/base/base_relu_max.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class ReLUMax : public OperatorOnCPU<base::ReLUMax> {
		public:
		    using self = ReLUMax;
			using supper = OperatorOnCPU<base::ReLUMax>;

            void relu_max(const Tensor &x, float max, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_RELU_MAX_H