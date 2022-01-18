#ifndef TENSORSTACK_KERNELS_CPU_PRELU_H
#define TENSORSTACK_KERNELS_CPU_PRELU_H

#include "backend/base/base_prelu.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class PReLU : public OperatorOnCPU<base::PReLU> {
		public:
		    using self = PReLU;
			using supper = OperatorOnCPU<base::PReLU>;

            void prelu(const Tensor &x, const Tensor &slope, int dim, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_PRELU_H