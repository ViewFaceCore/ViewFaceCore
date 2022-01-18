//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_DEVICE_CONTEXT_H
#define TENSORSTACK_DEVICE_CONTEXT_H


#include <global/hard_converter.h>
#include <global/device_admin.h>

#include "utils/ctxmgr_lite.h"

namespace ts {
    class DeviceHandle;
    class TS_DEBUG_API DeviceContext : public SetupContext<DeviceContext> {
    public:
        using self = DeviceContext;
        using shared = std::shared_ptr<self>;

        DeviceContext() = default;

        DeviceContext(ComputingDevice computing_device);

        ~DeviceContext();

        DeviceContext(const self &) = delete;
        self &operator=(const self &) = delete;

        void initialize(ComputingDevice computing_device);
        void finalize();

        void active();
        void synchronize();

        /**
         *
         * @param context new-context
         * @return return pre-device context
         * if new-context is not pre-context than synchronize the pre-context
         */
        static self* Switch(self *context);

        /**
         * pointing to device operating self-defined structure
         * not using in out scope
         */
        DeviceHandle *handle = nullptr;

        ComputingDevice computing_device;
        MemoryDevice memory_device;

    private:
        DeviceAdmin::function m_device_admin;

    public:
        DeviceContext(self &&other) {
            *this = std::move(other);
        }
        DeviceContext &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(handle);
            MOVE_MEMBER(computing_device);
            MOVE_MEMBER(memory_device);
            MOVE_MEMBER(m_device_admin);
#undef MOVE_MEMBER
            return *this;
        }
    };
}


#endif //TENSORSTACK_DEVICE_CONTEXT_H
