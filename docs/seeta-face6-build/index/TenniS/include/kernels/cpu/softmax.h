#ifndef TENSORSTACK_KERNELS_CPU_SOFTMAX_H
#define TENSORSTACK_KERNELS_CPU_SOFTMAX_H

#include "backend/base/base_softmax.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class Softmax : public OperatorOnCPU<base::Softmax> {
		public:
			using self = Softmax;
			using supper = OperatorOnCPU<base::Softmax>;

			void softmax(const Tensor &x, int dim, bool smooth, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_SOFTMAX_H