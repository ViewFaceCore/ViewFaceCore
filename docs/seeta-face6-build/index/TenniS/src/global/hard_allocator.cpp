//
// Created by lby on 2018/3/11.
//

#include "global/hard_allocator.h"
#include "utils/static.h"

#include <map>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "global/hard_converter.h"


namespace ts {
    static std::map<DeviceType, HardAllocator::function> &MapDeviceAllocator() {
        static std::map<DeviceType, HardAllocator::function> map_device_allocator;
        return map_device_allocator;
    };

    HardAllocator::function HardAllocator::Query(const DeviceType &device_type) TS_NOEXCEPT {
        auto &map_device_allocator = MapDeviceAllocator();
        auto device_allocator = map_device_allocator.find(device_type);
        if (device_allocator != map_device_allocator.end()) {
            return device_allocator->second;
        }
        return HardAllocator::function(nullptr);
    }

    void HardAllocator::Register(const DeviceType &device_type, const function &allocator) TS_NOEXCEPT {
        auto &map_device_allocator = MapDeviceAllocator();
        map_device_allocator[device_type] = allocator;
    }

    void HardAllocator::Clear() {
        auto &map_device_allocator = MapDeviceAllocator();
        map_device_allocator.clear();
    }

    void HardAllocator::RegisterV3(const DeviceType &device_type, const HardAllocator::_malloc &_new,
                                   const HardAllocator::_free &_delete,
                                   const HardAllocator::_realloc &_reset) TS_NOEXCEPT {
        Register(device_type, Bind(_new, _delete, _reset));
    }

    void HardAllocator::RegisterV2(const DeviceType &device_type, const HardAllocator::_malloc &_new,
                                   const HardAllocator::_free &_delete) TS_NOEXCEPT {
        auto _converter = HardConverter::Query(device_type, device_type);
        auto _reset = [_new, _delete, _converter](int id, size_t new_size, void *mem, size_t mem_size) -> void* {
            auto new_mem = _new(id, new_size);
            _converter(id, new_mem, id, mem, std::min(new_size, mem_size));
            _delete(id, mem);
            return new_mem;
        };
        RegisterV3(device_type, _new, _delete, _reset);
    }

    HardAllocator::function HardAllocator::Bind(const HardAllocator::_malloc &_new, const HardAllocator::_free &_delete,
                                                const HardAllocator::_realloc &_reset) {
        return [_new, _delete, _reset](int id, size_t new_size, void *mem, size_t mem_size) -> void * {
            if (new_size == 0) {
                _delete(id, mem);
                return nullptr;
            } else if (mem) {
                if (mem_size > 0) {
                    return _reset(id, new_size, mem, mem_size);
                } else {
                    _delete(id, mem);
                }
            }
            return _new(id, new_size);
        };
    }

    std::set<std::string> HardAllocator::AllKeys() TS_NOEXCEPT {
        auto &map_key_values = MapDeviceAllocator();
        std::set<std::string> keys;
        for (auto &key_value : map_key_values) { keys.insert(key_value.first.std()); }
        return keys;
    }
}
