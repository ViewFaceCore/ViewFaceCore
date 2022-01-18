//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_GLOBAL_CONVERTER_H
#define TENSORSTACK_GLOBAL_CONVERTER_H


#include "core/device.h"
#include "utils/except.h"
#include <functional>

#include <set>

namespace ts {
    class TS_DEBUG_API HardConverter {
    public:
        /**
         * HardConverter, convert memory between devices
         */
        using function = std::function<void(int, void *, int, const void *, size_t)>;

        /**
         * example of HardConverter
         * @param dst_id the dst device id
         * @param dst the dst memory pointer
         * @param src_id the src device id
         * @param src the dst memory pointer
         * @param size the copy size
         * @note the src and dst device type was specific given by register
         */
        void HardConverterFunction(int dst_id, void *dst, int src_id, const void *src, size_t size);

        /**
         * Query memory converter
         * @param dst_device_type querying dst device
         * @param src_device_type querying src device
         * @return converter
         * @note supporting called by threads without calling @sa RegisterConverter
         */
        static function Query(DeviceType dst_device_type, DeviceType src_device_type) TS_NOEXCEPT;

        /**
         * Register converter
         * @param dst_device_type registering dst device
         * @param src_device_type registering src device
         * @param converter registering converter
         * @note only can be called before running
         */
        static void Register(DeviceType dst_device_type, DeviceType src_device_type,
                             const function &converter) TS_NOEXCEPT;

        /**
         * No details for this API, so DO NOT call it
         */
        static void Clear();

        /**
         * @return set of pair of dst device and src device
         */
        static std::set<std::pair<std::string, std::string>> AllKeys() TS_NOEXCEPT;
    };
}

#endif //TENSORSTACK_GLOBAL_CONVERTER_H
