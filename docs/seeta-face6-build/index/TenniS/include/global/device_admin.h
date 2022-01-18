//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
#define TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H

#include <functional>

#include "core/device.h"

#include <set>

namespace ts {
    class DeviceHandle;
    class TS_DEBUG_API DeviceAdmin {
    public:

        /**
         * Action on device
         */
        enum Action {
            INITIALIZATION,  ///< initialize a handle on given device
            FINALIZATION,    ///< finalize a handle on given device
            ACTIVATION,      ///< call in using device
            SYNCHRONIZE,      ///< call in not using this action now
        };

        /**
         * DeviceAdmin, initialize or finalize device shang xia wen
         */
        using function = std::function<void(DeviceHandle * *, int, Action)>;


        /**
         * Example of DeviceAdmin
         * @param handle Pointer to the pointer of handle, ready to be initialized of finalized, which controlled by `action`
         * @param device_id admin device id
         * @param action @sa DeviceAction
         */
        void DeviceAdminFunction(DeviceHandle **handle, int device_id, Action action);

        /**
         * Query device admin
         * @param device_type querying computing device
         * @return DeviceAdmin
         * @note supporting called by threads without calling @sa RegisterDeviceAdmin
         * @note the query device should be computing device
         */
        static function Query(const DeviceType &device_type) TS_NOEXCEPT;

        /**
         * Register DeviceAdmin for specific device type
         * @param device_type specific @sa DeviceType
         * @param device_admin setting allocator
         * @note only can be called before running @sa QueryDeviceAdmin
         */
        static void Register(const DeviceType &device_type, const function &device_admin) TS_NOEXCEPT;

        /**
         * No details for this API, so DO NOT call it
         */
        static void Clear();

        /**
         * @return set of device type
         */
        static std::set<std::string> AllKeys() TS_NOEXCEPT;
    };
}


#endif //TENSORSTACK_GLOBAL_DEVICE_INITIALIZER_H
