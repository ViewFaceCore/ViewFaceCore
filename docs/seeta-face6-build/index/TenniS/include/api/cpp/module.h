//
// Created by kier on 2019/3/16.
//

#ifndef TENNIS_API_CPP_MODULE_H
#define TENNIS_API_CPP_MODULE_H

#include "../module.h"

#include "except.h"
#include "device.h"
#include "stream.h"

#include <string>

namespace ts {
    namespace api {
        /**
         * @see ts_SerializationFormat
         */
        class SerializationFormat {
        public:
            using self = SerializationFormat;

            SerializationFormat() : self(TS_BINARY) {}

            SerializationFormat(ts_SerializationFormat dtype) : raw(dtype) {}

            operator ts_SerializationFormat() const { return raw; }

            ts_SerializationFormat raw;
        };

        static const SerializationFormat BINARY = TS_BINARY;
        static const SerializationFormat TEXT = TS_TEXT;

        /**
         * @see ts_Module
         */
        class Module {
        public:
            using self = Module;
            using raw = ts_Module;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            static self NewRef(raw *ptr) { return self(ptr); }

            Module(const self &) = default;

            Module &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            bool operator==(std::nullptr_t) const { return get_raw() == nullptr; }

            bool operator!=(std::nullptr_t) const { return get_raw() != nullptr; }

            Module(std::nullptr_t) {}

            Module() = default;

            static Module Load(const std::string &path, SerializationFormat format = TS_BINARY) {
                Module loaded(ts_Module_Load(path.c_str(), ts_SerializationFormat(format)));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            static Module Load(StreamReader &stream, SerializationFormat format = TS_BINARY) {
                Module loaded(ts_Module_LoadFromStream(&stream, StreamReader::C, ts_SerializationFormat(format)));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            static Module Fusion(const Module &in, int32_t in_out_slot, const Module &out, int32_t out_in_slot) {
                Module fusion(ts_Module_Fusion(in.get_raw(), in_out_slot, out.get_raw(), out_in_slot));
                TS_API_AUTO_CHECK(fusion.m_impl != nullptr);
                return std::move(fusion);
            }

        private:
            Module(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_Module); }

            shared_raw m_impl;
        };
    }
}

#endif //TENNIS_API_CPP_MODULE_H
