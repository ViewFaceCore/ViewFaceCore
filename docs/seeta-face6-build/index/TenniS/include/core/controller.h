//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_CORE_CONTROLLER_H
#define TENSORSTACK_CORE_CONTROLLER_H

#include "memory.h"

#include <utils/api.h>

namespace ts {
    /**
     * MemoryController: Malloc memory and control them
     */
    class TS_DEBUG_API MemoryController {
    public:
        using self = MemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual ~MemoryController() = default;
        /**
         * alloc memory with size
         * @param size memory size (bytes)
         * @return allocated memory
         */
        virtual Memory alloc(size_t size) = 0;

        /**
         * Get memory size under control
         * @return
         */
        virtual uint64_t summary() const { return 0; };
    };

    class TS_DEBUG_API DynamicMemoryController : public MemoryController {
    public:
        using self = DynamicMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = MemoryController;
        /**
         * @param device the memory device
         */
        explicit DynamicMemoryController(const MemoryDevice &device);

        Memory alloc(size_t size) override;

    private:
        MemoryDevice m_device;
        HardAllocator::function m_allocator;
    };

    // TODO: add flow memory controller
}


#endif //TENSORSTACK_CORE_CONTROLLER_H
