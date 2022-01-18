#ifndef TENSORSTACK_KERNELS_CPU_TO_FLOAT_H
#define TENSORSTACK_KERNELS_CPU_TO_FLOAT_H

#include "backend/base/base_cast.h"


namespace ts {
	namespace cpu {
	    using ToFloat = OperatorOnAny<base::CastTo<FLOAT32>>;
	}
}


#endif //TENSORSTACK_KERNELS_CPU_TO_FLOAT_H