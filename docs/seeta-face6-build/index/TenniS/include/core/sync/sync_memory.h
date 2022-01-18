//
// Created by kier on 2018/12/26.
//

#ifndef TENSORSTACK_SYNC_SYNC_MEMORY_H
#define TENSORSTACK_SYNC_SYNC_MEMORY_H

#include <core/memory.h>
#include "sync_block.h"

#include <utils/api.h>

namespace ts {
    class TS_DEBUG_API SyncMemory {
    public:
        using self = SyncMemory;
        using shared = std::shared_ptr<self>;

        using Block = SyncBlock<MemoryDevice, Memory>;

        static Block::value_t dynamic_sync_handler(const Block::value_t &from_memory,
                                                   const Block::key_t &from_device,
                                                   const Block::key_t &to_device) {
            Memory to_memory(to_device, from_memory.size());
            memcpy(to_memory, from_memory);
            return to_memory;
        }

        SyncMemory(const Memory &memory, bool lock, Block::sync_handler handler) {
            m_sync_memory = std::make_shared<Block>(memory.device(), memory, handler, lock);
        }

        SyncMemory(const Memory &memory, bool lock = false)
            : SyncMemory(memory, lock, dynamic_sync_handler){}

        SyncMemory(const MemoryDevice &device, size_t size, bool lock = false)
            : SyncMemory(Memory(device, size), lock) {}

        SyncMemory(size_t size, bool lock = false)
                : SyncMemory(Memory(size), lock) {}

        SyncMemory(bool lock = false)
                : SyncMemory(0, lock) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(const HardMemory::shared &hard, size_t size, size_t shift = 0)
                : SyncMemory(Memory(hard, size, shift)) {}

        SyncMemory(const HardMemory::shared &hard, size_t size, size_t shift, bool lock)
                : SyncMemory(Memory(hard, size, shift), lock) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(HardMemory::shared &&hard, size_t size, size_t shift = 0)
                : SyncMemory(Memory(std::move(hard), size, shift)) {}

        SyncMemory(HardMemory::shared &&hard, size_t size, size_t shift, bool lock)
                : SyncMemory(Memory(std::move(hard), size, shift), lock) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(const HardMemory::shared &hard)
                : SyncMemory(Memory(hard)) {}

        SyncMemory(const HardMemory::shared &hard, bool lock)
                : SyncMemory(Memory(hard), lock) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(HardMemory::shared &&hard)
                : SyncMemory(Memory(std::move(hard))) {}

        SyncMemory(HardMemory::shared &&hard, bool lock)
                : SyncMemory(Memory(std::move(hard)), lock) {}

        /**
         * Moving constructed function
         * @param other other object
         */
        SyncMemory(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving assignment function
         * @param other other object
         */
        SyncMemory &operator=(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving constructed function
         * @param other other object
         */
        SyncMemory(self &&other) TS_NOEXCEPT {
            *this = std::move(other);
        }

        /**
         * Moving assignment function
         * @param other other object
         */
        SyncMemory &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_sync_memory);
#undef MOVE_MEMBER
            return *this;
        }

        /**
         * Swap to other object
         * @param other
         */
        void swap(self &other) {
            std::swap(this->m_sync_memory, other.m_sync_memory);
        }

        /**
         * Get size of memory
         * @return size of memory
         */
        size_t size() const { return m_sync_memory->value().size(); }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        void *data() {
            auto default_value = m_sync_memory->value();
            return default_value.data();
        }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        const void *data() const { return m_sync_memory->value().data(); }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        template<typename T>
        T *data() { return reinterpret_cast<T *>(this->data()); }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        template<typename T>
        const T *data() const { return reinterpret_cast<const T *>(this->data()); }

        /**
         * return Device of this memory
         * @return @see Device
         */
        const MemoryDevice &device() const { return  m_sync_memory->key(); }

        /**
         *
         * @return got weak memory
         */
        Memory weak_memory() const {
            return m_sync_memory->value().weak();
        }

        /**
         * got memory on device
         * @param device
         * @return
         */
        self view(const MemoryDevice &device) const {
            return self(m_sync_memory->view(device));
        }

        void broadcast() {
            m_sync_memory->broadcast();
        }

    private:
        SyncMemory(std::shared_ptr<Block> sync_memory) : m_sync_memory(std::move(sync_memory)) {}

        std::shared_ptr<Block> m_sync_memory;
    };
}


#endif //TENSORSTACK_SYNC_MEMORY_H
