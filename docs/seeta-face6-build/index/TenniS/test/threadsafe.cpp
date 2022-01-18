//
// Created by kier on 2019/2/14.
//

#include "utils/log.h"
#include "global/hard_allocator.h"
#include "core/memory.h"
#include "core/threadsafe/smart.h"

#define TS_LOG_CHECKING(condition) TS_LOG_INFO("Case [")((condition) ? "PASSED" : "FAILED")("]: ")(#condition)

using namespace ts;

static const std::string fake_device_type;

std::string p(void *p) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(sizeof(p) * 2) << std::setfill('0') << uint64_t(p);
    return oss.str();
}

void *fake_allocator(int id, size_t new_size, void *mem, size_t mem_size) {
    if (new_size == 0 && mem == nullptr) return nullptr;
    void *new_mem = nullptr;
    if (new_size == 0) {
        std::free(mem);
        TS_LOG_INFO << std::hex << "free(" << p(mem) << ")";
        return nullptr;
    } else if (mem != nullptr) {
        if (mem_size) {
            new_mem = std::realloc(mem, new_size);
        } else {
            std::free(mem);
            new_mem = std::malloc(new_size);
        }
        TS_LOG_INFO << std::hex << "realloc(" << p(mem) << ", " << std::oct << new_size << ") -> " << std::hex << "0x" << new_mem;
    } else {
        new_mem = std::malloc(new_size);
        TS_LOG_INFO << "malloc(" << new_size << ") -> " << std::hex << p(new_mem);
    }
    if (new_mem == nullptr) throw OutOfMemoryException(MemoryDevice(fake_device_type, id), new_size);
    return new_mem;
}

int main() {
    using namespace ts;

    HardAllocator::Register(fake_device_type, fake_allocator);

    using TSMemory = Smart<Memory>;


    MemoryDevice fake_device(fake_device_type, 0);
    TSMemory a(Memory(fake_device, 100));

    TS_LOG_CHECKING(a.use_count() == 1);

    TSMemory b = std::move(a);

    TSMemory c = b.weak();
    TSMemory d = c;

    TS_LOG_CHECKING(a.use_count() == 0);
    TS_LOG_CHECKING(b.use_count() == 1);
    TS_LOG_CHECKING(c.use_count() == 1);
    TS_LOG_CHECKING(d.use_count() == 1);

    TSMemory e = d.strong();

    TS_LOG_CHECKING(e.use_count() == 2);
}
