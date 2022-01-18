#ifndef TENSORSTACK_KERNELS_CPU_PREWHITEN_H
#define TENSORSTACK_KERNELS_CPU_PREWHITEN_H

#include "backend/base/base_prewhiten.h"
#include "operator_on_cpu.h"

namespace ts {
	namespace cpu {
		class PreWhiten : public OperatorOnCPU<base::PreWhiten> {
		public:
		    using self = PreWhiten;
			using supper = OperatorOnCPU<base::PreWhiten>;

			void prewhiten(const Tensor &x, Tensor &out) override;
		};
	}
}



#endif //TENSORSTACK_KERNELS_CPU_PREWHITEN_H