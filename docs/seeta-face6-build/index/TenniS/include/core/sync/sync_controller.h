//
// Created by kier on 2018/12/26.
//

#ifndef TENSORSTACK_SYNC_SYNC_CONTROLLER_H
#define TENSORSTACK_SYNC_SYNC_CONTROLLER_H

#include <memory>
#include <core/device.h>
#include <core/memory.h>
#include <core/controller.h>

#include "sync_block.h"
#include "sync_memory.h"

#include <utils/api.h>

namespace ts {
    class TS_DEBUG_API SyncMemoryController {
    public:
        using self = SyncMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual ~SyncMemoryController() = default;

        /**
         * alloc memory with size
         * @param size memory size (bytes)
         * @return allocated memory
         * The default device should control in other way
         */
        virtual SyncMemory alloc(size_t size) = 0;

        /**
         * alloc memory with size
         * @param device device to get memory
         * @param size memory size (bytes)
         * @return allocated memory
         */
        virtual SyncMemory alloc(const MemoryDevice &device, size_t size) = 0;

        virtual SyncMemoryController::shared clone() const = 0;

        virtual std::string summary() const { return "{}"; }
    };

    class TS_DEBUG_API SyncDeviceMemoryController : public SyncMemoryController {
    public:
        using self = SyncDeviceMemoryController;
        using supper = SyncMemoryController;

        using shared = std::shared_ptr<self>;  ///< smart pointer

        SyncDeviceMemoryController(const MemoryDevice &device) : m_device(device) {}

        SyncMemory alloc(size_t size) override {
            return this->alloc(m_device, size);
        }

        SyncMemory alloc(const MemoryDevice &device, size_t size) override = 0;

    protected:
        MemoryDevice m_device;
    };

    template <typename _MemoryController>
    class TS_DEBUG_API HypeSyncMemoryController
            : public SyncDeviceMemoryController,
            public std::enable_shared_from_this<HypeSyncMemoryController<_MemoryController>> {
    public:
        using self = HypeSyncMemoryController;
        using supper = SyncDeviceMemoryController;

        using shared = std::shared_ptr<self>;

        using BaseMemoryController = _MemoryController;

        static shared Make(const MemoryDevice &device, bool need_lock = false) {
            return shared(new self(device, need_lock));
        }

    private:
        HypeSyncMemoryController(const MemoryDevice &device, bool need_lock = false)
                : SyncDeviceMemoryController(device)
                , m_sync_controllers(device, std::make_shared<BaseMemoryController>(device), sync_controller_handler, need_lock)
                , m_memory_need_lock(need_lock) {
        }

    public:
        void clear(const MemoryDevice &device) {
            m_sync_controllers.clear(device);
        }

        SyncMemory alloc(size_t size) override {
            return this->alloc(m_device, size);
        }

        SyncMemory alloc(const MemoryDevice &device, size_t size) override {
            auto controller = m_sync_controllers.sync(device);
            auto memory = controller->alloc(size);
            return SyncMemory(memory, m_memory_need_lock, this->sync_handler());
        }

        SyncMemoryController::shared clone() const override {
            return shared(new self(m_device, m_memory_need_lock));
        }

        SyncMemory::Block::sync_handler sync_handler() {
            auto shared_this = this->shared_from_this();
            return [=](const typename SyncMemory::Block::value_t &from_memory,
                       const typename SyncMemory::Block::key_t &from_device,
                       const typename SyncMemory::Block::key_t &to_device) {
                auto controller = shared_this->m_sync_controllers.sync(to_device);
                auto to_memory = controller->alloc(from_memory.size());
                memcpy(to_memory, from_memory);
                return to_memory;
            };
        }

        std::string summary() const override {
            std::ostringstream oss;
            oss << "{";
            bool comma = false;
            m_sync_controllers.foreach([&](
                    const typename SyncControllerBlock::key_t &device,
                    const typename SyncControllerBlock::value_t &controller){
                if (comma) oss << ", ";
                else comma = true;
                oss << "\"" << device << "\": \"" << memory_size_string(controller->summary()) << "\"";
            });
            oss << "}";
            return oss.str();
        }

    private:
        using SyncControllerBlock = SyncBlock<MemoryDevice, std::shared_ptr<BaseMemoryController>>;

        SyncControllerBlock m_sync_controllers;

        bool m_memory_need_lock;

        static typename SyncControllerBlock::value_t sync_controller_handler(
                const typename SyncControllerBlock::value_t &,
                const typename SyncControllerBlock::key_t &,
                const typename SyncControllerBlock::key_t &device) {
            return std::make_shared<BaseMemoryController>(device);
        }
    };

    using DynamicSyncMemoryController = HypeSyncMemoryController<DynamicMemoryController>;
}


#endif //TENSORSTACK_SYNC_SYNC_CONTROLLER_H
