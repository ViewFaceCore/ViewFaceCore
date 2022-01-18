//
// Created by lby on 2018/3/11.
//


#include "core/controller.h"

#include "utils/assert.h"

#include "utils/ctxmgr_lite_support.h"

namespace ts {

    DynamicMemoryController::DynamicMemoryController(const MemoryDevice &device)
            : m_device(device) {
        m_allocator = HardAllocator::Query(device.type());
        TS_CHECK(m_allocator != nullptr) << "Can not found memory controller for " << device.type() << eject;
    }

    Memory DynamicMemoryController::alloc(size_t size) {
        return Memory(std::make_shared<HardMemory>(m_device, m_allocator, size));
    }
}

// TS_LITE_CONTEXT(ts::MemoryController)
