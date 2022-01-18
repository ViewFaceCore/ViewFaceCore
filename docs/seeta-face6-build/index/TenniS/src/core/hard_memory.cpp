//
// Created by lby on 2018/2/11.
//

#include "core/hard_memory.h"

#include <cstdlib>
#include <utility>


#include "utils/assert.h"

namespace ts {
    HardMemory::HardMemory(const MemoryDevice &device)
            : m_device(device) {
        m_allocator = HardAllocator::Query(device.type());
        TS_AUTO_CHECK(m_allocator != nullptr);
    }

    HardMemory::HardMemory(const MemoryDevice &device, size_t size)
            : HardMemory(device) {
        this->resize(size);
    }

    HardMemory::HardMemory(const MemoryDevice &device, const HardAllocator::function &allocator)
        : m_device(device), m_allocator(allocator){
        TS_AUTO_CHECK(m_allocator != nullptr);
    }

    HardMemory::HardMemory(const MemoryDevice &device, const HardAllocator::function &allocator, size_t size)
            : HardMemory(device, allocator) {
        this->resize(size);
    }

    HardMemory::HardMemory(const MemoryDevice &device, void *data, size_t size)
            : m_device(device), m_capacity(size), m_data(data) {
    }

    HardMemory::~HardMemory() {
        if (m_allocator) m_allocator(m_device.id(), 0, m_data, 0);
    }

    void HardMemory::dispose() {
        if (m_allocator) m_allocator(m_device.id(), 0, m_data, 0);
        m_data = nullptr;
    }

    void HardMemory::expect(size_t size) {
        if (!m_allocator) TS_LOG_ERROR("Borrowed memory can not be expected.") << eject;
        if (size > m_capacity) {
            m_data = m_allocator(m_device.id(), size, m_data, m_capacity);
            m_capacity = size;
        }
    }

    void HardMemory::shrink(size_t size) {
        if (!m_allocator) TS_LOG_ERROR("Borrowed memory can not be shrunk.") << eject;
        if (size < m_capacity) {
            m_data = m_allocator(m_device.id(), size, m_data, m_capacity);
            m_capacity = size;
        }
    }

    void HardMemory::resize(size_t size) {
        if (!m_allocator) TS_LOG_ERROR("Borrowed memory can not be resized.") << eject;
        if (size != m_capacity) {
            m_data = m_allocator(m_device.id(), size, m_data, 0);
            m_capacity = size;
        }
    }

    void HardMemory::swap(self &other) {
        std::swap(this->m_device, other.m_device);
        std::swap(this->m_capacity, other.m_capacity);
        std::swap(this->m_data, other.m_data);
        std::swap(this->m_allocator, other.m_allocator);
    }

	HardMemory::HardMemory(self &&other) TS_NOEXCEPT{
        *this = std::move(other);
    }

    HardMemory &HardMemory::operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
        MOVE_MEMBER(m_device);
        MOVE_MEMBER(m_capacity);
        MOVE_MEMBER(m_data);
        MOVE_MEMBER(m_allocator);
#undef MOVE_MEMBER
        return *this;
    }
}
