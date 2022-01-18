//
// Created by lby on 2018/2/11.
//

#ifndef TENSORSTACK_CORE_HARD_MEMORY_H
#define TENSORSTACK_CORE_HARD_MEMORY_H

#include <cstddef>

#include "device.h"
#include "global/hard_allocator.h"

namespace ts {
    /**
     * Hardware memory
     */
    class TS_DEBUG_API HardMemory {
    public:
        using self = HardMemory;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        HardMemory(const self &) = delete;

        const HardMemory &operator=(const self &) = delete;

        /**
         * Initialize hardware memory
         * @param device memory @sa Device
         */
        explicit HardMemory(const MemoryDevice &device);

        /**
         * Initialize hardware memory
         * @param device memory @sa Device
         * @param size expected size
         */
        explicit HardMemory(const MemoryDevice &device, size_t size);

        /**
         * Initialize hardware memory
         * @param device memory @sa Device
         * @param allocator memory allocator @see HardAllocator
         */
        explicit HardMemory(const MemoryDevice &device, const HardAllocator::function &allocator);

        /**
         * Initialize hardware memory
         * @param device memory @sa Device
         * @param allocator memory allocator @see HardAllocator
         * @param size expected size
         */
        explicit HardMemory(const MemoryDevice &device, const HardAllocator::function &allocator, size_t size);

        /**
         * Initialize hardware memory
         * @param device memory @sa Device
         * @param data borrowed memory pointer
         * @param size borrowed memory size
         */
        explicit HardMemory(const MemoryDevice &device, void *data, size_t size);

        ~HardMemory();

        /**
         * Moving constructed function
         * @param other other memory
         */
        HardMemory(self &&other) TS_NOEXCEPT;

        /**
         * Moving assignment function
         * @param other other memory
         */
		HardMemory &operator=(self &&other) TS_NOEXCEPT;

        /**
         * Swap to other object
         * @param other
         */
        void swap(self &other);

        /**
         * Dispose all hardware memory
         */
        void dispose();

        /**
         * expend memory size to param size
         * @param size expected size
         */
        void expect(size_t size);

        /**
         * shrink memory size to param size
         * @param size expected size
         */
        void shrink(size_t size);

        /**
         * shrink resize size to param size
         * @param size expected size
         */
        void resize(size_t size);

        /**
         * Runing deivce
         * @return running @sa Device
         */
        const MemoryDevice &device() const { return m_device; }

        /**
         * Get memory capacity on hardware
         * @return return memory capacity
         */
        size_t capacity() const { return m_capacity; }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        void *data() { return m_data; }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        const void *data() const { return m_data; }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        template<typename T>
        T *data() { return reinterpret_cast<T *>(this->data()); }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        template<typename T>
        const T *data() const { return reinterpret_cast<const T *>(this->data()); }

    private:
        MemoryDevice m_device;                         ///< running device
        size_t m_capacity = 0;                   ///< memory capacity
        void *m_data = nullptr;                ///< memory start pointer
        HardAllocator::function m_allocator = nullptr;    ///< memory allocatorx
    };

    /**
     * Swap two objects
     * @param mem1 first object
     * @param mem2 second object
     */
    inline void swap(HardMemory &mem1, HardMemory &mem2) { mem1.swap(mem2); }
}


#endif //TENSORSTACK_CORE_HARD_MEMORY_H
