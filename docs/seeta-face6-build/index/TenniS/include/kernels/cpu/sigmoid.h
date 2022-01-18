#ifndef TENSORSTACK_KERNELS_CPU_SIGMOID_H
#define TENSORSTACK_KERNELS_CPU_SIGMOID_H

#include "backend/base/base_sigmoid.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class Sigmoid : public OperatorOnCPU<base::Sigmoid> {
		public:
		    using self = Sigmoid;
			using supper = OperatorOnCPU<base::Sigmoid>;

            void active(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_SIGMOID_H