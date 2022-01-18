//
// Created by kier on 2018/12/19.
//

#ifndef TENSORSTACK_CORE_SINK_H
#define TENSORSTACK_CORE_SINK_H

#include <cstdint>
#include <iostream>

// #define TS_SINK_CHECKING_OVERFLOW

namespace ts {
    using top_number = int64_t;

    template <typename _T>
    class number {
    public:
        using T = _T;
        _T value;

        number() = default;
        number(_T value) : value(value) {}

        explicit operator _T() const { return value; }

        static _T Max();

        static _T Min();
    };

#define TS_DEFINE_NUMBER_MAX(_T, value) template <> _T number<_T>::Max() { return (value); }
#define TS_DEFINE_NUMBER_MIN(_T, value) template <> _T number<_T>::Min() { return (value); }

    TS_DEFINE_NUMBER_MAX(int8_t, INT8_MAX)
    TS_DEFINE_NUMBER_MIN(int8_t, INT8_MIN)
    TS_DEFINE_NUMBER_MAX(int16_t, INT16_MAX)
    TS_DEFINE_NUMBER_MIN(int16_t, INT16_MIN)
    TS_DEFINE_NUMBER_MAX(int32_t, INT32_MAX)
    TS_DEFINE_NUMBER_MIN(int32_t, INT32_MIN)
    TS_DEFINE_NUMBER_MAX(int64_t, INT64_MAX)
    TS_DEFINE_NUMBER_MIN(int64_t, INT64_MIN)
#undef TS_DEFINE_NUMBER_MAX
#undef TS_DEFINE_NUMBER_MIN

    template <typename _T>
    class _signed {
    public:
        using T = _T;
    };

#define TS_DEFINE_SIGNED(_UNSIGNED_T, _SIGNED) \
    template <> class _signed<_UNSIGNED_T> { public:using T = _SIGNED ; };

    TS_DEFINE_SIGNED(uint8_t, int8_t)
    TS_DEFINE_SIGNED(uint16_t, int16_t)
    TS_DEFINE_SIGNED(uint32_t, int32_t)
    TS_DEFINE_SIGNED(uint64_t, int64_t)
#undef TS_DEFINE_SIGNED

    template <typename _T>
    class _double_width {
    public:
        using T = _T;
    };
#define TS_DEFINE_DOUBLE_WIDTH(_T, _2T) \
    template <> class _double_width<_T> { public:using T = _2T ; };

    TS_DEFINE_DOUBLE_WIDTH(int8_t, int16_t)
    TS_DEFINE_DOUBLE_WIDTH(int16_t, int32_t)
    TS_DEFINE_DOUBLE_WIDTH(int32_t, int64_t)
    TS_DEFINE_DOUBLE_WIDTH(int64_t, long long)
    TS_DEFINE_DOUBLE_WIDTH(uint8_t, uint16_t)
    TS_DEFINE_DOUBLE_WIDTH(uint16_t, uint32_t)
    TS_DEFINE_DOUBLE_WIDTH(uint32_t, uint64_t)
    TS_DEFINE_DOUBLE_WIDTH(uint64_t, unsigned long long)
#undef TS_DEFINE_DOUBLE_WIDTH

    template <typename _COMMON_T, typename _FIXED_T>
    class number_converter {
    public:
        static _FIXED_T ToFixed(_COMMON_T value, int _Q) {
            // TODO: supporting round number
            auto fixed = top_number(value * (1 << _Q));
#ifndef TS_SINK_CHECKING_OVERFLOW
            return _FIXED_T(fixed);
#else
            auto max = top_number(number<_FIXED_T>::Max());
            auto min = top_number(number<_FIXED_T>::Min());
            return _FIXED_T(fixed > max ? max : (fixed < min ? min : fixed));
#endif
        }

        static _COMMON_T ToCommon(_FIXED_T value, int _Q) {
            return _COMMON_T(value) / (1 << _Q);
        }
    };

#ifndef TS_SINK_CHECKING_OVERFLOW
#define TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(_T) \
    template <typename _FIXED_T> \
    class number_converter<_T, _FIXED_T> { \
    public: \
        using _COMMON_T = _T; \
        static _FIXED_T ToFixed(_COMMON_T value, int _Q) { \
            auto fixed = top_number(value) << _Q; \
            return _FIXED_T(fixed); \
        } \
        static _COMMON_T ToCommon(_FIXED_T value, int _Q) { \
            return _COMMON_T(value >> _Q); \
        } \
    };
#else
#define TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(_T) \
    template <typename _FIXED_T> \
    class number_converter<_T, _FIXED_T> { \
    public: \
        using _COMMON_T = _T; \
        static _FIXED_T ToFixed(_COMMON_T value, int _Q) { \
            auto fixed = top_number(value) << _Q; \
            auto max = top_number(number<_FIXED_T>::Max()); \
            auto min = top_number(number<_FIXED_T>::Min()); \
            return _FIXED_T(fixed > max ? max : (fixed < min ? min : fixed)); \
        } \
        static _COMMON_T ToCommon(_FIXED_T value, int _Q) { \
            return _COMMON_T(value >> _Q); \
        } \
    };
#endif

    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(int8_t)
    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(int16_t)
    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(int32_t)
    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(int64_t)
    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(uint8_t)
    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(uint16_t)
    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(uint32_t)
    TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER(uint64_t)
#undef TS_DEFINE_FIXED_NUMBER_INTEGER_CONVERTER

#define TS_DEFINE_FIXED_NUMBER_UNSIGNED_CONVERTER(_T) \
    template <typename _COMMON_T> \
    class number_converter<_COMMON_T, _T> { \
    public: \
        using _FIXED_T = _T; \
        using _SIGNED_FIXED_T = typename _signed<_FIXED_T>::T; \
        static _FIXED_T ToFixed(_COMMON_T value, int _Q) { \
            return _FIXED_T(number_converter<_COMMON_T, _SIGNED_FIXED_T>::ToFixed(value, _Q)); \
        } \
        static _COMMON_T ToCommon(_FIXED_T value, int _Q) { \
            return number_converter<_COMMON_T, _SIGNED_FIXED_T>::ToCommon(_SIGNED_FIXED_T(value), _Q); \
        } \
    };

    TS_DEFINE_FIXED_NUMBER_UNSIGNED_CONVERTER(uint8_t)
    TS_DEFINE_FIXED_NUMBER_UNSIGNED_CONVERTER(uint16_t)
    TS_DEFINE_FIXED_NUMBER_UNSIGNED_CONVERTER(uint32_t)
    TS_DEFINE_FIXED_NUMBER_UNSIGNED_CONVERTER(uint64_t)
#undef TS_DEFINE_FIXED_NUMBER_UNSIGNED_CONVERTER

    template <typename _T, int _Q>
    class fixed_point_number : public number<_T> {
    public:
        using self = fixed_point_number;
        using supper = number<_T>;

        using T = _T;
        using S = typename _signed<_T>::T;

        static const int Q = _Q;

        static self Max() { return from_fixed(_T(number<S>::Max())); }
        static self Min() { return from_fixed(_T(number<S>::Min())); }

        fixed_point_number() = default;

#define TS_DECLARE_CONSTRUCTOR(_COMMON_T) \
        fixed_point_number(_COMMON_T value) : supper(_T(number_converter<decltype(value), S>::ToFixed(value, _Q))) {}

        TS_DECLARE_CONSTRUCTOR(int8_t)
        TS_DECLARE_CONSTRUCTOR(int16_t)
        TS_DECLARE_CONSTRUCTOR(int32_t)
        TS_DECLARE_CONSTRUCTOR(int64_t)
        TS_DECLARE_CONSTRUCTOR(uint8_t)
        TS_DECLARE_CONSTRUCTOR(uint16_t)
        TS_DECLARE_CONSTRUCTOR(uint32_t)
        TS_DECLARE_CONSTRUCTOR(uint64_t)
        TS_DECLARE_CONSTRUCTOR(float)
        TS_DECLARE_CONSTRUCTOR(double)
#undef TS_DECLARE_CONSTRUCTOR

#define TS_DECLARE_OPERATOR(_COMMON_T) \
        operator _COMMON_T() const { return number_converter<_COMMON_T, S>::ToCommon(S(this->value), _Q); }

        TS_DECLARE_OPERATOR(int8_t)
        TS_DECLARE_OPERATOR(int16_t)
        TS_DECLARE_OPERATOR(int32_t)
        TS_DECLARE_OPERATOR(int64_t)
        TS_DECLARE_OPERATOR(uint8_t)
        TS_DECLARE_OPERATOR(uint16_t)
        TS_DECLARE_OPERATOR(uint32_t)
        TS_DECLARE_OPERATOR(uint64_t)
        TS_DECLARE_OPERATOR(float)
        TS_DECLARE_OPERATOR(double)
#undef TS_DECLARE_OPERATOR

        template <typename _TO_T>
        _TO_T to() const { return _TO_T(*this); }

        static self from_fixed(_T fixed) {
            self result;
            result.value = fixed;
            return result;
        }
    };

    template <typename _T, int _Q>
    using sink = fixed_point_number<_T, _Q>;

    template <int _Q>
    using sink8 = sink<uint8_t, _Q>;

    template <int _Q>
    using sink16 = sink<uint16_t, _Q>;

    template <int _Q>
    using sink32 = sink<uint32_t, _Q>;

    template <int _Q>
    using sink64 = sink<uint64_t, _Q>;

    template <typename _T, int _Q>
    std::ostream &operator<<(std::ostream &out, const sink<_T, _Q> &s) {
        return out << double(s);
    }

    template <typename _T, int _Q>
    std::istream &operator>>(std::istream &in, sink<_T, _Q> &s) {
        double d;
        in >> d;
        s = d;
        return in;
    }

    template <typename _T, int _Q>
    sink<_T, _Q> operator-(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) {
        using _S = typename _signed<_T>::T;
        using _DS = typename _double_width<_S>::T;
        auto fixed = _DS(lhs.value) - _DS(rhs.value);
#ifndef TS_SINK_CHECKING_OVERFLOW
        return sink<_T, _Q>::from_fixed(_T(fixed));
#else
        auto max = top_number(number<_S>::Max());
        auto min = top_number(number<_S>::Min());
        return sink<_T, _Q>::from_fixed(_T(fixed > max ? max : (fixed < min ? min : fixed)));
#endif
    }

    template <typename _T, int _Q>
    sink<_T, _Q> operator+(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) {
        using _S = typename _signed<_T>::T;
        using _DS = typename _double_width<_S>::T;
        auto fixed = _DS(lhs.value) + _DS(rhs.value);
#ifndef TS_SINK_CHECKING_OVERFLOW
        return sink<_T, _Q>::from_fixed(_T(fixed));
#else
        auto max = top_number(number<_S>::Max());
        auto min = top_number(number<_S>::Min());
        return sink<_T, _Q>::from_fixed(_T(fixed > max ? max : (fixed < min ? min : fixed)));
#endif
    }

    template <typename _T, int _Q>
    sink<_T, _Q> operator-(const sink<_T, _Q> &lhs) {
        return sink<_T, _Q>::from_fixed(-lhs.value);
    }

    template <typename _T, int _Q>
    sink<_T, _Q> operator*(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) {
        using _S = typename _signed<_T>::T;
        using _DS = typename _double_width<_S>::T;
        auto fixed = ((_DS)(lhs.value) * (_DS)(rhs.value)) >> ((_Q) + (_Q) - (_Q));
#ifndef TS_SINK_CHECKING_OVERFLOW
        return sink<_T, _Q>::from_fixed(_T(fixed));
#else
        auto max = top_number(number<_S>::Max());
        auto min = top_number(number<_S>::Min());
        return sink<_T, _Q>::from_fixed(_T(fixed > max ? max : (fixed < min ? min : fixed)));
#endif
    }

    template <typename _T, int _Q>
    sink<_T, _Q> operator/(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) {
        using _S = typename _signed<_T>::T;
        using _DS = typename _double_width<_S>::T;
        auto fixed = rhs.value == 0 ? sink<_T, _Q>::Max().value : (((_DS)lhs.value) << ((_Q) + (_Q) - (_Q))) / rhs.value;
#ifndef TS_SINK_CHECKING_OVERFLOW
        return sink<_T, _Q>::from_fixed(_T(fixed));
#else
        auto max = top_number(number<_S>::Max());
        auto min = top_number(number<_S>::Min());
        return sink<_T, _Q>::from_fixed(_T(fixed > max ? max : (fixed < min ? min : fixed)));
#endif
    }

    template <typename _T, int _Q>
    bool operator<(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) { return lhs.value < rhs.value; }

    template <typename _T, int _Q>
    bool operator>(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) { return lhs.value > rhs.value; }

    template <typename _T, int _Q>
    bool operator<=(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) { return lhs.value <= rhs.value; }

    template <typename _T, int _Q>
    bool operator>=(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) { return lhs.value >= rhs.value; }

    template <typename _T, int _Q>
    bool operator==(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) { return lhs.value == rhs.value; }

    template <typename _T, int _Q>
    bool operator!=(const sink<_T, _Q> &lhs, const sink<_T, _Q> &rhs) { return lhs.value != rhs.value; }

    // TODO: add operators on sink type, inv
}

#endif //TENSORSTACK_CORE_SINK_H
