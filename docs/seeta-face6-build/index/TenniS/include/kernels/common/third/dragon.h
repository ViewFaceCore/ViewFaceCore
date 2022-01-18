//
// Created by kier on 2019/9/5.
//

#ifndef TENSORSTACK_THIRD_DRAGON_H
#define TENSORSTACK_THIRD_DRAGON_H

#include "core/tensor.h"
#include "core/tensor_builder.h"

#include <set>
#include <numeric>
#include <runtime/operator.h>
#include <frontend/intime.h>

#include "dragon/context_cpu.h"
#include "dragon/context_cuda.h"
#include "dragon/workspace.h"
#include "dragon/type_meta.h"
#include "dragon/tensor.h"
#include "dragon/operator.h"
#include "dragon/op_kernel.h"
#include "dragon/logging.h"

namespace ts {
    namespace dragon {
        using std::vector;
        using std::string;
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_H
