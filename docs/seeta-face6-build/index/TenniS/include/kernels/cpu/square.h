#ifndef TENSORSTACK_KERNELS_CPU_SQUARE_H
#define TENSORSTACK_KERNELS_CPU_SQUARE_H

#include "backend/base/base_activation.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class Square : public OperatorOnCPU<base::Activation> {
		public:
		    using self = Square;
   	            using supper = OperatorOnCPU<base::Activation>;

            void active(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_SQUARE_H
