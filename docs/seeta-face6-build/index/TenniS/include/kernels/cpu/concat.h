#ifndef TENSORSTACK_KERNELS_CPU_CONCAT_H
#define TENSORSTACK_KERNELS_CPU_CONCAT_H

#include "operator_on_cpu.h"
#include "backend/base/base_concat.h"


namespace ts {
	namespace cpu {
		class Concat : public OperatorOnAny<base::Concat> {
		public:
			using self = Concat;
			using supper = OperatorOnAny<base::Concat>;

			Concat() = default;

			void concat(const std::vector<Tensor> &x, int dim, Tensor &out) override;
		};
	}
}


#endif //TENSORSTACK_KERNELS_CPU_CONCAT_H