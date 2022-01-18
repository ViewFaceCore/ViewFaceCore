//
// Created by kier on 2018/11/2.
//

#include "global/device_admin.h"

#include <map>
#include <cstdlib>
#include <iostream>

namespace ts {
    static std::map<DeviceType, DeviceAdmin::function> &MapDeviceTypeAdmin() {
        static std::map<DeviceType, DeviceAdmin::function> map_device_type_admin;
        return map_device_type_admin;
    };

    DeviceAdmin::function DeviceAdmin::Query(const DeviceType &device_type) TS_NOEXCEPT {
        auto &map_device_type_admin = MapDeviceTypeAdmin();
        auto device_type_admin = map_device_type_admin.find(device_type);
        if (device_type_admin != map_device_type_admin.end()) {
            return device_type_admin->second;
        }
        return DeviceAdmin::function(nullptr);
    }

    void DeviceAdmin::Register(const DeviceType &device_type, const function &device_admin) TS_NOEXCEPT {
        auto &map_device_type_admin = MapDeviceTypeAdmin();
        map_device_type_admin[device_type] = device_admin;
    }

    void DeviceAdmin::Clear() {
        auto &map_device_type_admin = MapDeviceTypeAdmin();
        map_device_type_admin.clear();
    }

    std::set<std::string> DeviceAdmin::AllKeys() TS_NOEXCEPT {
        auto &map_key_values = MapDeviceTypeAdmin();
        std::set<std::string> keys;
        for (auto &key_value : map_key_values) { keys.insert(key_value.first.std()); }
        return keys;
    }
}
