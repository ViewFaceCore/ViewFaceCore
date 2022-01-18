#ifndef TENSORSTACK_KERNELS_CPU_UNSQUEEZE_H
#define TENSORSTACK_KERNELS_CPU_UNSQUEEZE_H

#include "backend/base/base_unsqueeze.h"


namespace ts {
	namespace cpu {
	    using Unsqueeze = OperatorOnAny<base::Unsqueeze>;
	}
}


#endif //TENSORSTACK_KERNELS_CPU_UNSQUEEZE_H