//
// Created by lby on 2018/3/11.
//


#include <algorithm>
#include <core/memory.h>


#include "core/memory.h"
#include "global/hard_converter.h"
#include "utils/assert.h"

namespace ts {
    static void *const FakeUsagePtr = (void *) (0x19910929);

    static void default_usage_destructor(void *) {}

    static std::shared_ptr<void> default_usage() {
        return std::shared_ptr<void>(FakeUsagePtr, default_usage_destructor);
    }

    Memory::Memory(const HardMemory::shared &hard, size_t size, size_t shift)
            : m_hard(hard), m_size(size), m_shift(shift), m_usage(default_usage()) {
    }

    Memory::Memory(HardMemory::shared &&hard, size_t size, size_t shift)
            : m_hard(std::move(hard)), m_size(size), m_shift(shift), m_usage(default_usage()) {
    }

    Memory::Memory(const MemoryDevice &device, size_t size)
            : m_hard(new HardMemory(device, size)), m_size(size), m_shift(0), m_usage(default_usage()) {
    }

    Memory::Memory(size_t size)
            : m_hard(new HardMemory(MemoryDevice(), size)), m_size(size), m_shift(0), m_usage(default_usage()) {
    }

    void Memory::destructor(const std::function<void(void *)> &dtor, void *data) {
        m_usage.reset(data, dtor);
    }

    void Memory::destructor(const std::function<void(void)> &dtor) {
        m_usage.reset(FakeUsagePtr, [dtor](void *) -> void { dtor(); });
    }

    void Memory::swap(Memory::self &other) {
        std::swap(this->m_hard, other.m_hard);
        std::swap(this->m_size, other.m_size);
        std::swap(this->m_shift, other.m_shift);
        std::swap(this->m_usage, other.m_usage);
    }

    Memory::Memory(Memory::self &&other) TS_NOEXCEPT {
        *this = std::move(other);
    }

    Memory &Memory::operator=(Memory::self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
        MOVE_MEMBER(m_hard);
        MOVE_MEMBER(m_size);
        MOVE_MEMBER(m_shift);
        MOVE_MEMBER(m_usage);
#undef MOVE_MEMBER
        return *this;
    }

    Memory::Memory(const HardMemory::shared &hard)
            : Memory(hard, hard->capacity()) {
    }

    Memory::Memory(HardMemory::shared &&hard)
            : Memory(std::move(hard), hard->capacity()) {
    }

    long Memory::use_count() const {
        return m_usage.use_count();
    }

    const MemoryDevice &Memory::device() const {
        return this->m_hard->device();
    }

    Memory::Memory()
            : Memory(0) {
    }

    Memory::Memory(const MemoryDevice &device, void *data, size_t size) {
        m_size = size;
        m_shift = 0;
        m_hard = std::make_shared<HardMemory>(device, data, size);
    }

    Memory Memory::weak() const {
        return Memory(device(), const_cast<void *>(this->data()), m_size);
    }

    void memcpy(Memory &dst, const Memory &src, size_t size) {
        TS_AUTO_CHECK(dst.size() >= size);
        TS_AUTO_CHECK(src.size() >= size);
        HardConverter::function converter = HardConverter::Query(dst.device().type(), src.device().type());
        TS_AUTO_CHECK(converter != nullptr);
        converter(dst.device().id(), dst.data(), src.device().id(), src.data(), size);
    }

    void memcpy(Memory &dst, const Memory &src) {
        TS_AUTO_CHECK(dst.size() >= src.size());
        auto size = src.size();
        HardConverter::function converter = HardConverter::Query(dst.device().type(), src.device().type());
        TS_AUTO_CHECK(converter != nullptr);
        converter(dst.device().id(), dst.data(), src.device().id(), src.data(), size);
    }

    size_t
    memcpy(void *dst_ptr, const Device &dst_device, size_t dst_size, const void *src_ptr, const Device &src_device,
           size_t src_size) {
        auto copy_size = std::min(src_size, dst_size);
        HardConverter::function converter = HardConverter::Query(dst_device.type(), src_device.type());
        TS_AUTO_CHECK(converter != nullptr);
        converter(dst_device.id(), dst_ptr, src_device.id(), src_ptr, copy_size);
        return copy_size;
    }

    void memset(void *dst_ptr, const Device &dst_device, size_t dst_size, const void *src_ptr, const Device &src_device,
                size_t src_size) {
        HardConverter::function cross_device_converter = HardConverter::Query(dst_device.type(),
                                                                              src_device.type());
        TS_AUTO_CHECK(cross_device_converter != nullptr);
        HardConverter::function in_device_converter = (dst_device == src_device) ? cross_device_converter
                                                                                 : HardConverter::Query(
                        dst_device.type(), dst_device.type());
        TS_AUTO_CHECK(in_device_converter != nullptr);
        cross_device_converter(dst_device.id(), dst_ptr, src_device.id(), src_ptr, std::min(dst_size, src_size));
        size_t copy_anchor = src_size;
        while (copy_anchor <= size_t(dst_size >> 1)) {
            in_device_converter(dst_device.id(), reinterpret_cast<char *>(dst_ptr) + copy_anchor, dst_device.id(),
                                dst_ptr, copy_anchor);
            copy_anchor <<= 1;
        }
        if (dst_size > copy_anchor) {
            in_device_converter(dst_device.id(), reinterpret_cast<char *>(dst_ptr) + copy_anchor, dst_device.id(),
                                dst_ptr, dst_size - copy_anchor);
        }
    }
}