//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_OP_KERNEL_H
#define TENSORSTACK_THIRD_DRAGON_OP_KERNEL_H

#include "utils/log.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/tensor_builder.h"
#include "frontend/intime.h"

#include "type_meta.h"

namespace ts {
    namespace dragon {
        namespace kernel {
            template<typename A, typename B, typename Context>
            inline void TypeA2B(size_t count, const A *src, B *dst, Context *ctx) {
                auto memory_device = ctx->memory_device();
                ts::Tensor SRC(ts::Memory(memory_device, const_cast<A *>(src), count * sizeof(A)),
                               ts::Tensor::Prototype(dtypeid<A>::id, {int(count)}));
                ts::Tensor DST;
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    DST = tensor::cast(dtypeid<B>::id, SRC);
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    DST = intime::cast(SRC, dtypeid<B>::id);
                }
                ts::memcpy(dst, memory_device, count * sizeof(B),
                           DST.data(), memory_device, count * sizeof(B));
            }
        }
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_OP_KERNEL_H
