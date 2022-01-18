//
// Created by kier on 2019/3/16.
//

#ifndef TENNIS_API_CPP_DTYPE_H
#define TENNIS_API_CPP_DTYPE_H

#include "../tensor.h"

namespace ts {
    namespace api {
        /**
         * @see ts_DTYPE
         */
        class DTYPE {
        public:
            using self = DTYPE;

            DTYPE() : self(TS_VOID) {}

            DTYPE(ts_DTYPE dtype) : raw(dtype) {}

            operator ts_DTYPE() const { return raw; }

            ts_DTYPE raw;
        };

        static const DTYPE VOID = TS_VOID;
        static const DTYPE INT8 = TS_INT8;
        static const DTYPE UINT8 = TS_UINT8;
        static const DTYPE INT16 = TS_INT16;
        static const DTYPE UINT16 = TS_UINT16;
        static const DTYPE INT32 = TS_INT32;
        static const DTYPE UINT32 = TS_UINT32;
        static const DTYPE INT64 = TS_INT64;
        static const DTYPE UINT64 = TS_UINT64;
        static const DTYPE FLOAT32 = TS_FLOAT32;
        static const DTYPE FLOAT64 = TS_FLOAT64;
        static const DTYPE CHAR8 = TS_CHAR8;

        /**
         * Get dtype width
         * @param dtype in ts_DTYPE
         * @return type width
         */
        inline int type_bytes(DTYPE dtype) {
            switch (dtype.raw) {
                case TS_VOID: return 0;
                case TS_INT8: return 1;
                case TS_UINT8: return 1;
                case TS_INT16: return 2;
                case TS_UINT16: return 2;
                case TS_INT32: return 4;
                case TS_UINT32: return 4;
                case TS_INT64: return 8;
                case TS_UINT64: return 8;
                case TS_FLOAT32: return 4;
                case TS_FLOAT64: return 8;
                case TS_CHAR8: return 1;
            }
            return 0;
        }

        /**
         * Get dtype description string
         * @param dtype in ts_DTYPE
         * @return string description
         */
        inline const char *type_str(DTYPE dtype) {
            switch (dtype.raw) {
                case TS_VOID: return "void";
                case TS_INT8: return "int8";
                case TS_UINT8: return "uint8";
                case TS_INT16: return "int64";
                case TS_UINT16: return "uint64";
                case TS_INT32: return "int32";
                case TS_UINT32: return "uint32";
                case TS_INT64: return "int64";
                case TS_UINT64: return "uint64";
                case TS_FLOAT32: return "float32";
                case TS_FLOAT64: return "float64";
                case TS_CHAR8: return "char8";
            }
            return "unknown";
        }

        template <ts_DTYPE T> struct dtype { using declare = void; };

        template <> struct dtype<TS_VOID> { using declare = void; };
        template <> struct dtype<TS_INT8> { using declare = int8_t; };
        template <> struct dtype<TS_UINT8> { using declare = uint8_t; };
        template <> struct dtype<TS_INT16> { using declare = int16_t; };
        template <> struct dtype<TS_UINT16> { using declare = uint16_t; };
        template <> struct dtype<TS_INT32> { using declare = int32_t; };
        template <> struct dtype<TS_UINT32> { using declare = uint32_t; };
        template <> struct dtype<TS_INT64> { using declare = int64_t; };
        template <> struct dtype<TS_UINT64> { using declare = uint64_t; };
        template <> struct dtype<TS_FLOAT32> { using declare = float; };
        template <> struct dtype<TS_FLOAT64> { using declare = double; };
        template <> struct dtype<TS_CHAR8> { using declare = char; };

        template <typename T> struct dtypeid { static const ts_DTYPE id = TS_VOID; };

        template <> struct dtypeid<void> { static const ts_DTYPE id = TS_VOID; };
        template <> struct dtypeid<int8_t> { static const ts_DTYPE id = TS_INT8; };
        template <> struct dtypeid<uint8_t> { static const ts_DTYPE id = TS_UINT8; };
        template <> struct dtypeid<int16_t> { static const ts_DTYPE id = TS_INT16; };
        template <> struct dtypeid<uint16_t> { static const ts_DTYPE id = TS_UINT16; };
        template <> struct dtypeid<int32_t> { static const ts_DTYPE id = TS_INT32; };
        template <> struct dtypeid<uint32_t> { static const ts_DTYPE id = TS_UINT32; };
        template <> struct dtypeid<int64_t> { static const ts_DTYPE id = TS_INT64; };
        template <> struct dtypeid<uint64_t> { static const ts_DTYPE id = TS_UINT64; };
        template <> struct dtypeid<float> { static const ts_DTYPE id = TS_FLOAT32; };
        template <> struct dtypeid<double> { static const ts_DTYPE id = TS_FLOAT64; };
        template <> struct dtypeid<char> { static const ts_DTYPE id = TS_CHAR8; };
    }
}

#endif //TENNIS_API_CPP_DTYPE_H
