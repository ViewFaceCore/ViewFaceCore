//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_CORE_MEMORY_H
#define TENSORSTACK_CORE_MEMORY_H

#include "device.h"
#include "hard_memory.h"

#include <memory>

namespace ts {

    /**
     * Memory, directly memory on specific device
     */
    class TS_DEBUG_API Memory {
    public:
        using self = Memory;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(const HardMemory::shared &hard, size_t size, size_t shift = 0);

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(HardMemory::shared &&hard, size_t size, size_t shift = 0);

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(const HardMemory::shared &hard);

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(HardMemory::shared &&hard);

        /**
         * Initialize Memory
         * @param device memory device
         * @param size sizeof this memory block
         */
        Memory(const MemoryDevice &device, size_t size);

        /**
         * Initialize Memory
         * @param device memory device
         * @param data borrowed memory
         * @param size sizeof this memory block
         */
        Memory(const MemoryDevice &device, void *data, size_t size);

        /**
         * Initialize Memory, with cpu zero memory
         * @param size sizeof the memory block
         */
        explicit Memory(size_t size);

        Memory();

        /**
         * Moving constructed function
         * @param other other object
         */
        Memory(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving assignment function
         * @param other other object
         */
        Memory &operator=(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving constructed function
         * @param other other object
         */
        Memory(self &&other) TS_NOEXCEPT;

        /**
         * Moving assignment function
         * @param other other object
         */
        Memory &operator=(self &&other) TS_NOEXCEPT;

        /**
         * Swap to other object
         * @param other
         */
        void swap(self &other);

        /**
         * Get size of memory
         * @return size of memory
         */
        size_t size() const { return m_size; }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        void *data() { return m_hard->data<char>() + m_shift; }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        const void *data() const { return m_hard->data<char>() + m_shift; }

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
         * Set callback when memory will be free
         * @param dtor destructor
         * @param data param will pass to destructor
         * @note the use_count will reset after this API
         * @note use one of use_count or this API to control memory, do not use them both
         */
        void destructor(const std::function<void(void *)> &dtor, void *data);

        /**
         * Set callback when memory will be free
         * @param dtor destructor
         * @param data param will pass to destructor
         * @note the use_count will reset after this API
         * @note use one of use_count or this API to control memory, do not use them both
         */
        void destructor(const std::function<void(void)> &dtor);

        /**
         * return use count of this memory block
         * @return use count
         */
        long use_count() const;

        /**
         * return Device of this memory
         * @return @see Device
         */
        const MemoryDevice &device() const;

        /**
         * got weak reference, it will be invalid after this memory deleted
         * @return weak memory
         */
        Memory weak() const;

    private:
        HardMemory::shared m_hard = nullptr;  ///< hardware memory
        size_t m_size = 0;                              ///< sizeof this memory block
        size_t m_shift = 0;                             ///< shift from start pointer
        std::shared_ptr<void> m_usage = nullptr;      ///< for memory usage count
    };

    /**
     * Swap two objects
     * @param obj1 first object
     * @param obj2 second object
     */
    inline void swap(Memory &obj1, Memory &obj2) { obj1.swap(obj2); }

    /**
     * copy memory in device or cross devices
     * @param dst the dst memory
     * @param src the src memory
     * @param size copy size
     */
    TS_DEBUG_API void memcpy(Memory &dst, const Memory &src, size_t size);

    /**
     * copy memory in device or cross devices, copy src size
     * @param dst the dst memory
     * @param src the src memory
     */
    TS_DEBUG_API void memcpy(Memory &dst, const Memory &src);

    /**
     * copy memory in device or cross devices
     * @param dst_ptr the dst memory
     * @param dst_device the dst memory device
     * @param dst_size the dst memory size
     * @param src_ptr the src memory
     * @param src_device the src memory device
     * @param src_size the src memory size
     * @return really copy size
     */
    TS_DEBUG_API size_t memcpy(void *dst_ptr, const Device &dst_device, size_t dst_size,
                  const void *src_ptr, const Device &src_device, size_t src_size);

    /**
     * set memory in device or cross devices
     * @param dst_ptr the dst memory
     * @param dst_device the dst memory device
     * @param dst_size the dst memory size
     * @param src_ptr the src memory
     * @param src_device the src memory device
     * @param src_size the src memory size
     */
    TS_DEBUG_API void memset(void *dst_ptr, const Device &dst_device, size_t dst_size,
                const void *src_ptr, const Device &src_device, size_t src_size);
}


#endif //TENSORSTACK_CORE_MEMORY_H
