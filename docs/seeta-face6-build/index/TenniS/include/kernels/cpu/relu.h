#ifndef TENSORSTACK_KERNELS_CPU_RELU_H
#define TENSORSTACK_KERNELS_CPU_RELU_H

#include "backend/base/base_relu.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class ReLU : public OperatorOnCPU<base::ReLU> {
		public:
		    using self = ReLU;
			using supper = ts::Operator;

            void active(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_PRELU_H