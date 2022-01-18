//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_TYPE_META_H
#define TENSORSTACK_THIRD_DRAGON_TYPE_META_H

#include "context_cpu.h"
#include "context_cuda.h"

namespace ts {
    namespace dragon {
        namespace TypeMeta {
            template<typename T>
            struct _ID {
                static const int id = 0;
            };
            template<>
            struct _ID<CPUContext> {
                static const int id = 1;
            };
            template<>
            struct _ID<CUDAContext> {
                static const int id = 2;
            };

            template<typename T>
            static int Id() { return _ID<T>::id; }
        };

        inline std::string TypeMetaToString(const ts::Tensor::Prototype &proto) {
            std::ostringstream oss;
            oss << proto;
            return oss.str();
        }
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_TYPE_META_H
