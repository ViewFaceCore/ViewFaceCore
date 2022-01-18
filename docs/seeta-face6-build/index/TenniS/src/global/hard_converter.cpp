//
// Created by lby on 2018/3/11.
//

#include "global/hard_converter.h"
#include "utils/static.h"

#include <map>

#include <iostream>

namespace ts {
    static std::map<DeviceType, std::map<DeviceType, HardConverter::function>> &MapDstSrcConverter() {
        static std::map<DeviceType, std::map<DeviceType, HardConverter::function>> map_dst_src_converter;
        return map_dst_src_converter;
    };

    HardConverter::function
    HardConverter::Query(DeviceType dst_device_type, DeviceType src_device_type) TS_NOEXCEPT {
        auto &map_dst_src_converter = MapDstSrcConverter();
        auto dst_src_converter = map_dst_src_converter.find(dst_device_type);
        if (dst_src_converter != map_dst_src_converter.end()) {
            auto &map_src_converter = dst_src_converter->second;
            auto src_converter = map_src_converter.find(src_device_type);
            if (src_converter != map_src_converter.end()) {
                return src_converter->second;
            }
        }
        return HardConverter::function(nullptr);
    }

    void
    HardConverter::Register(DeviceType dst_device_type, DeviceType src_device_type,
                            const function &converter) TS_NOEXCEPT {
        auto &map_dst_src_converter = MapDstSrcConverter();
        auto dst_src_converter = map_dst_src_converter.find(dst_device_type);
        if (dst_src_converter != map_dst_src_converter.end()) {
            auto &map_src_converter = dst_src_converter->second;
            map_src_converter[src_device_type] = converter;
        } else {
            std::map<DeviceType, HardConverter::function> map_src_converter;
            map_src_converter.insert(std::make_pair(src_device_type, converter));
            map_dst_src_converter.insert(std::make_pair(dst_device_type, std::move(map_src_converter)));
        }
    }

    void HardConverter::Clear() {
        auto &map_dst_src_converter = MapDstSrcConverter();
        map_dst_src_converter.clear();
    }

    std::set<std::pair<std::string, std::string>> HardConverter::AllKeys() TS_NOEXCEPT {
        auto &map_dst_src_converter = MapDstSrcConverter();
        std::set<std::pair<std::string, std::string>> keys;
        for (auto &dst_src_converter : map_dst_src_converter) {
            auto dst = dst_src_converter.first;
            for (auto &src_converter : dst_src_converter.second) {
                auto src = src_converter.first;
                keys.insert(std::make_pair(dst.std(), src.std()));
            }
        }
        return keys;
    }
}
