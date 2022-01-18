//
// Created by kier on 2019/1/8.
//

#ifndef TENSORSTACK_MEMORY_FLOW_H
#define TENSORSTACK_MEMORY_FLOW_H

#include <utils/implement.h>
#include "core/controller.h"


namespace ts {
    class TS_DEBUG_API QueuedStackMemoryController : public MemoryController {
    public:
        using self = QueuedStackMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = MemoryController;
        /**
         * @param device the memory device
         */
        // explicit QueuedStackMemoryController(const MemoryDevice &device);

        // Memory alloc(size_t size) override;
    };


    class TS_DEBUG_API VatMemoryController : public MemoryController {
    public:
        using self = VatMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = MemoryController;
        /**
         * @param device the memory device
         */
        explicit VatMemoryController(const MemoryDevice &device);

        ~VatMemoryController() override;

        Memory alloc(size_t size) override;

        uint64_t summary() const override ;

    private:
        class Implement;
        Declare<Implement> m_impl;
    };


    class TS_DEBUG_API StackMemoryController : public MemoryController {
    public:
        using self = VatMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = MemoryController;
        /**
         * @param device the memory device
         */
        explicit StackMemoryController(const MemoryDevice &device);

        Memory alloc(size_t size) override;

    private:
        class Implement;
        Declare<Implement> m_impl;
    };


    using FlowMemoryController = VatMemoryController;
}


#endif //TENSORSTACK_MEMORY_FLOW_H
