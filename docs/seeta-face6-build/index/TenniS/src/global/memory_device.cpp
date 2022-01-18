//
// Created by kier on 2018/5/19.
//

#include "global/memory_device.h"
#include <map>

namespace ts {

    static std::map<DeviceType, DeviceType> &MapMemoryDevice() {
        static std::map<DeviceType, DeviceType> map_memory_device;
        return map_memory_device;
    };

    DeviceType ComputingMemory::Query(const DeviceType &compute_device_type) {
        auto &map_memory_device = MapMemoryDevice();
        auto memory_device = map_memory_device.find(compute_device_type);
        if (memory_device != map_memory_device.end()) {
            return memory_device->second;
        }
        throw NoMemoryDeviceException(compute_device_type);
    }

    MemoryDevice ComputingMemory::Query(const Device &compute_device) {
        return MemoryDevice(Query(compute_device.type()), compute_device.id());
    }

    void ComputingMemory::Register(const DeviceType &compute_device_type,
                                   const DeviceType &memory_device_type) TS_NOEXCEPT {
        auto &map_memory_device = MapMemoryDevice();
        map_memory_device[compute_device_type] = memory_device_type;
    }

    void ComputingMemory::Clear() {
        auto &map_memory_device = MapMemoryDevice();
        map_memory_device.clear();
    }

    std::set<std::pair<std::string, std::string>> ComputingMemory::AllItems() TS_NOEXCEPT {
        auto &map_memory_device = MapMemoryDevice();
        std::set<std::pair<std::string, std::string>> items;
        for (auto &pair_compute_memory : map_memory_device) {
            items.insert(std::make_pair(pair_compute_memory.first.std(), pair_compute_memory.second.std()));
        }
        return items;
    }
}