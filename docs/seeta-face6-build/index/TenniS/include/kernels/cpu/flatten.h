#ifndef TENSORSTACK_KERNELS_CPU_FLATTEN_H
#define TENSORSTACK_KERNELS_CPU_FLATTEN_H

#include "backend/base/base_flatten.h"


namespace ts {
	namespace cpu {
	    using Flatten = OperatorOnAny<base::Flatten>;
	}
}


#endif //TENSORSTACK_KERNELS_CPU_FLATTEN_H