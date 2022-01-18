//
// Created by kier on 2019/1/8.
//

#include <memory/flow.h>
#include <global/hard_converter.h>

#include "memory/flow.h"

#include "utils/assert.h"
#include "orz/vat.h"

#include <list>

namespace ts {
    class VatMemoryController::Implement {
    public:
        using self = Implement;
        MemoryDevice m_device;
        HardAllocator::function m_managed_allocator;
        std::shared_ptr<Vat> m_vat;
    };

    VatMemoryController::VatMemoryController(const MemoryDevice &device) {
        TS_AUTO_CHECK(m_impl.get() != nullptr);
        auto hard_allocator = HardAllocator::Query(device.type());
        TS_CHECK(hard_allocator != nullptr) << "Can not found memory controller for " << device.type();
        using namespace std::placeholders;
        auto hard_free = std::bind(hard_allocator, device.id(), 0, _1, 0);
        auto pot_allocator = [hard_allocator, device, hard_free](size_t size) -> std::shared_ptr<void> {
            return std::shared_ptr<void>(hard_allocator(device.id(), size, nullptr, 0), hard_free);
        };

        m_impl->m_device = device;
        m_impl->m_vat = std::make_shared<Vat>(pot_allocator);
        auto &vat = m_impl->m_vat;
        m_impl->m_managed_allocator = [vat](int, size_t new_size, void *mem, size_t mem_size) -> void * {
            void *new_mem = nullptr;
            if (new_size == 0) {
                // TS_LOG_DEBUG << "free(" << mem << ")";
                vat->free(mem);
                return nullptr;
            } else if (mem != nullptr) {
                if (mem_size > 0) {
                    TS_LOG_ERROR << "Reach the un-given code" << eject;
                } else {
                    vat->free(mem);
                    new_mem = vat->malloc(new_size);
                }
                // TS_LOG_DEBUG << "realloc(" << mem << ") -> " << new_mem;
            } else {
                new_mem = vat->malloc(new_size);
                // TS_LOG_DEBUG << "malloc() -> " << new_mem;
            }
            return new_mem;
        };
    }

    VatMemoryController::~VatMemoryController() {
        m_impl->m_vat->deprecated();
    }

    Memory VatMemoryController::alloc(size_t size) {
        return Memory(std::make_shared<HardMemory>(m_impl->m_device, m_impl->m_managed_allocator, size));
    }

    uint64_t VatMemoryController::summary() const {
        return m_impl->m_vat->summary();
    }

    class StackMemoryBlock {
    public:
        using self = StackMemoryBlock;

        bool used = true;

        uint64_t start = 0;
        uint64_t size = 0;

        self *prev = nullptr;
        self *next = nullptr;
    };

    class StackMemoryList {
    public:
        using self = StackMemoryList;
        using Block = StackMemoryBlock;

        StackMemoryList() {
            m_head.next = &m_tail;
            m_tail.prev = &m_head;
        };

        bool empty() const {
            return m_head.next == &m_tail;
        }

        Block *alloc(size_t size) {
            if (this->empty()) {
                return new_on_next(&m_head, size);
            } else if (this->m_tail.prev->start >= this->m_head.next->start) {
                return new_on_next(m_tail.prev, size);
            } else {

            }
            return nullptr;
        }

        void free(Block *block) {
            block->used = false;
            while (true) {
                if (block->used) break;
                if (!block->next->used) {
                    block->next->start -= block->size;
                    block->next->prev = block->prev;
                    block->prev->next = block->next;
                    auto next_block = block->next;
                    delete block;
                    block = next_block;
                    continue;
                }
                if (!block->prev->used) {
                    block->prev->size += block->size;
                    block->prev->next = block->next;
                    block->next->prev = block->prev;
                    auto next_block = block->prev;
                    delete block;
                    block = next_block;
                    continue;
                }
                block->next->prev = block->prev;
                block->prev->next = block->next;
                delete block;
            }
        }

    private:
        Block m_head;
        Block m_tail;

        Block *new_on_next(Block *block, size_t size) {
            auto new_block = new Block;
            new_block->start = block->start + block->size;
            new_block->size = uint64_t(size);
            new_block->used = true;
            auto block_prev = block;
            auto block_next = block->next;

            new_block->prev = block_prev;
            new_block->next = block_next;

            block_prev->next = new_block;
            block_next->prev = new_block;

            return new_block;
        }
    };

    class StackMemoryController::Implement {
    public:
        using self = Implement;

        Implement(const MemoryDevice &device) {
            m_memory = std::make_shared<HardMemory>(device);
        }

        StackMemoryBlock alloc_block(size_t size);
        void free_block(StackMemoryBlock block);

    private:
        std::list<StackMemoryBlock> m_map;
        std::shared_ptr<HardMemory> m_memory;
    };

    StackMemoryController::StackMemoryController(const MemoryDevice &device)
            : m_impl(device) {

    }

    Memory StackMemoryController::alloc(size_t size) {
        return Memory();
    }
}