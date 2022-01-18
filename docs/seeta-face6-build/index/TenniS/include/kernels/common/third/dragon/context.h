//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_CONTEXT_H
#define TENSORSTACK_THIRD_DRAGON_CONTEXT_H

#include <cstdint>
#include <cstdlib>

#include "core/memory.h"

namespace ts {
    namespace dragon {
        class Workspace;

        class BaseContext {
        public:
            using self = BaseContext;

            BaseContext(Workspace *ws);

            void set_stream_id(int id);

            const ComputingDevice &computing_device() const { return m_computing_device; }

            const MemoryDevice &memory_device() const { return m_memory_device; }

        private:
            ComputingDevice m_computing_device;
            MemoryDevice m_memory_device;
        };
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_CONTEXT_H
