#include "kernels/cpu/memory_cpu.h"

#include "utils/static.h"

#include "global/hard_allocator.h"
#include "global/hard_converter.h"
#include "global/memory_device.h"

#include "utils/assert.h"

#include <cstring>

namespace ts {
    void *cpu_allocator(int id, size_t new_size, void *mem, size_t mem_size) {
        if (new_size == 0 && mem == nullptr) return nullptr;
        void *new_mem = nullptr;
        if (new_size == 0) {
            std::free(mem);
            return nullptr;
        } else if (mem != nullptr) {
            if (mem_size) {
                new_mem = std::realloc(mem, new_size);
            } else {
                std::free(mem);
                new_mem = std::malloc(new_size);
            }
        } else {
            new_mem = std::malloc(new_size);
        }
        if (new_mem == nullptr) throw OutOfMemoryException(MemoryDevice(CPU, id), new_size);
        return new_mem;
    }

    void
    cpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        TS_UNUSED(dst_id);
        TS_UNUSED(src_id);
        std::memcpy(dst, src, size);
    }
}

TS_STATIC_ACTION(ts::HardAllocator::Register, ts::CPU, ts::cpu_allocator)

TS_STATIC_ACTION(ts::HardConverter::Register, ts::CPU, ts::CPU, ts::cpu_converter)

TS_STATIC_ACTION(ts::ComputingMemory::Register, ts::CPU, ts::CPU)