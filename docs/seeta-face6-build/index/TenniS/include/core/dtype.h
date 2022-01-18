//
// Created by kier on 2018/5/25.
//

#ifndef TENSORSTACK_CORE_TYPE_H
#define TENSORSTACK_CORE_TYPE_H

#include <cstdint>

namespace ts {
    enum DTYPE {
        VOID        = 0,
        INT8        = 1,
        UINT8       = 2,
        INT16       = 3,
        UINT16      = 4,
        INT32       = 5,
        UINT32      = 6,
        INT64       = 7,
        UINT64      = 8,
        FLOAT16     = 9,
        FLOAT32     = 10,
        FLOAT64     = 11,
        PTR         = 12,              ///< for ptr type, with length of sizeof(void*) bytes
        CHAR8       = 13,            ///< for char saving string
        CHAR16      = 14,           ///< for char saving utf-16 string
        CHAR32      = 15,           ///< for char saving utf-32 string
        UNKNOWN8    = 16,        ///< for self define type, with length of 1 byte
        UNKNOWN16   = 17,
        UNKNOWN32   = 18,
        UNKNOWN64   = 19,
        UNKNOWN128  = 20,

        BOOLEAN     = 21,    // bool type, using byte in native
        COMPLEX32   = 22,  // complex 32(16 + 16)
        COMPLEX64   = 23,  // complex 64(32 + 32)
        COMPLEX128  = 24,  // complex 128(64 + 64)

        SINK8Q0     = 25,
        SINK8Q1     = 26,
        SINK8Q2     = 27,
        SINK8Q3     = 28,
        SINK8Q4     = 29,
        SINK8Q5     = 30,
        SINK8Q6     = 31,
        SINK8Q7     = 32,
    };

    inline int type_bytes(DTYPE dtype) {
        static const auto FakeUsagePtr = (void *) (0x19910929);
        switch (dtype) {
            case VOID: return 0;
            case INT8: return 1;
            case UINT8: return 1;
            case INT16: return 2;
            case UINT16: return 2;
            case INT32: return 4;
            case UINT32: return 4;
            case INT64: return 8;
            case UINT64: return 8;
            case FLOAT16: return 2;
            case FLOAT32: return 4;
            case FLOAT64: return 8;
            case PTR: return sizeof(FakeUsagePtr);
            case CHAR8: return 1;
            case CHAR16: return 2;
            case CHAR32: return 4;
            case UNKNOWN8: return 1;
            case UNKNOWN16: return 2;
            case UNKNOWN32: return 4;
            case UNKNOWN64: return 8;
            case UNKNOWN128: return 16;
            case BOOLEAN: return 1;
            case COMPLEX32: return 4;
            case COMPLEX64: return 8;
            case COMPLEX128: return 16;
            case SINK8Q0: return 1;
            case SINK8Q1: return 1;
            case SINK8Q2: return 1;
            case SINK8Q3: return 1;
            case SINK8Q4: return 1;
            case SINK8Q5: return 1;
            case SINK8Q6: return 1;
            case SINK8Q7: return 1;
        }
        return 0;
    }

    inline const char *type_str(DTYPE dtype) {
        switch (dtype) {
            case VOID: return "void";
            case INT8: return "int8";
            case UINT8: return "uint8";
            case INT16: return "int64";
            case UINT16: return "uint64";
            case INT32: return "int32";
            case UINT32: return "uint32";
            case INT64: return "int64";
            case UINT64: return "uint64";
            case FLOAT16: return "float16";
            case FLOAT32: return "float32";
            case FLOAT64: return "float64";
            case PTR: return "pointer";
            case CHAR8: return "char8";
            case CHAR16: return "char16";
            case CHAR32: return "char32";
            case UNKNOWN8: return "unknown8";
            case UNKNOWN16: return "unknown16";
            case UNKNOWN32: return "unknown32";
            case UNKNOWN64: return "unknown64";
            case UNKNOWN128: return "unknown128";
            case BOOLEAN: return "bool";
            case COMPLEX32: return "complex32";
            case COMPLEX64: return "complex64";
            case COMPLEX128: return "complex128";
            case SINK8Q0: return "sink8q0";
            case SINK8Q1: return "sink8q1";
            case SINK8Q2: return "sink8q2";
            case SINK8Q3: return "sink8q3";
            case SINK8Q4: return "sink8q4";
            case SINK8Q5: return "sink8q5";
            case SINK8Q6: return "sink8q6";
            case SINK8Q7: return "sink8q7";
        }
        return "unknown";
    }

    template <DTYPE T> struct dtype { using declare = void; };

    template <> struct dtype<VOID> { using declare = void; };
    template <> struct dtype<INT8> { using declare = int8_t; };
    template <> struct dtype<UINT8> { using declare = uint8_t; };
    template <> struct dtype<INT16> { using declare = int16_t; };
    template <> struct dtype<UINT16> { using declare = uint16_t; };
    template <> struct dtype<INT32> { using declare = int32_t; };
    template <> struct dtype<UINT32> { using declare = uint32_t; };
    template <> struct dtype<INT64> { using declare = int64_t; };
    template <> struct dtype<UINT64> { using declare = uint64_t; };
    template <> struct dtype<FLOAT32> { using declare = float; };
    template <> struct dtype<FLOAT64> { using declare = double; };
    template <> struct dtype<PTR> { using declare = void*; };
    template <> struct dtype<CHAR8> { using declare = char; };
    template <> struct dtype<CHAR16> { using declare = char16_t; };
    template <> struct dtype<CHAR32> { using declare = char32_t; };
    template <> struct dtype<BOOLEAN> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q0> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q1> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q2> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q3> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q4> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q5> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q6> { using declare = uint8_t; };
    template <> struct dtype<SINK8Q7> { using declare = uint8_t; };

    template <typename T> struct dtypeid { static const DTYPE id = VOID; };

    template <> struct dtypeid<void> { static const DTYPE id = VOID; };
    template <> struct dtypeid<int8_t> { static const DTYPE id = INT8; };
    template <> struct dtypeid<uint8_t> { static const DTYPE id = UINT8; };
    template <> struct dtypeid<int16_t> { static const DTYPE id = INT16; };
    template <> struct dtypeid<uint16_t> { static const DTYPE id = UINT16; };
    template <> struct dtypeid<int32_t> { static const DTYPE id = INT32; };
    template <> struct dtypeid<uint32_t> { static const DTYPE id = UINT32; };
    template <> struct dtypeid<int64_t> { static const DTYPE id = INT64; };
    template <> struct dtypeid<uint64_t> { static const DTYPE id = UINT64; };
    template <> struct dtypeid<float> { static const DTYPE id = FLOAT32; };
    template <> struct dtypeid<double> { static const DTYPE id = FLOAT64; };
    template <> struct dtypeid<void*> { static const DTYPE id = PTR; };
    template <> struct dtypeid<char> { static const DTYPE id = CHAR8; };
}


#endif //TENSORSTACK_CORE_TYPE_H
