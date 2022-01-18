//
// Created by kier on 2019/2/12.
//

#ifndef TENSORSTACK_CORE_IEEE754_FLOAT_H
#define TENSORSTACK_CORE_IEEE754_FLOAT_H

#include <cstddef>
#include <cmath>
#include <cstdint>

#include "utils/except.h"

#include "dtype.h"
#include "tensor_builder.h"

namespace ts {

    template<size_t _W>
    class bitset_type {
    };

    template<>
    class bitset_type<8> {
    public:
        using type = uint8_t;
        using signed_type = int8_t ;
    };

    template<>
    class bitset_type<16> {
    public:
        using type = uint16_t;
        using signed_type = int16_t ;
    };

    template<>
    class bitset_type<32> {
    public:
        using type = uint32_t;
        using signed_type = int32_t ;
    };

    template<>
    class bitset_type<64> {
    public:
        using type = uint64_t;
        using signed_type = int64_t ;
    };

    template<size_t _W>
    class bitset_code {
    public:
        using self = bitset_code;
        using type = typename bitset_type<_W>::type;

        type code;

        bitset_code() : code(0) {}

        explicit bitset_code(type value) : code(value) {}

        explicit operator type() const { return code; }
    };

    template<size_t _W>
    class upper_float {
    };

    template<>
    class upper_float<8> {
    public:
        using type = float;
    };

    template<>
    class upper_float<16> {
    public:
        using type = float;
    };

    template<>
    class upper_float<32> {
    public:
        using type = float;
    };

    template<>
    class upper_float<64> {
    public:
        using type = double;
    };

    template<size_t _W, size_t _SIGN, size_t _EXPONENT, size_t _FRACTION>
    class ieee754_float {
    public:
        constexpr static const auto W = _W;
        constexpr static const auto sign = _SIGN;
        constexpr static const auto exponent = _EXPONENT;
        constexpr static const auto fraction = _FRACTION;
        constexpr static const auto bias = (1 << (exponent - 1)) - 1;

        using self = ieee754_float;
        using bitset = bitset_code<W>;

        template<size_t _T_W, size_t _T_SIGN, size_t _T_EXPONENT, size_t _T_FRACTION>
        friend
        class ieee754_float;

        ieee754_float(const bitset &bits) : m_bits(bits) {}

        ieee754_float(const self &other) : m_bits(other.m_bits) {}

        template<size_t _T_W, size_t _T_SIGN, size_t _T_EXPONENT, size_t _T_FRACTION>
        explicit ieee754_float(const ieee754_float<_T_W, _T_SIGN, _T_EXPONENT, _T_FRACTION> &other) {
            // for float 0
            if ((other.code() & ~other.mask_sign().code) == 0) {
                this->m_bits.code = typename bitset::type((other.value_sign() << (W - 1)));
                return;
            }

            auto shift_left = int64_t(fraction) - int64_t(other.fraction);
            auto part_sign = (other.value_sign() << (W - 1));
            auto part_fraction = shift_left > 0
                                 ? (other.value_fraction() << shift_left)
                                 : (other.value_fraction() >> -shift_left);
            auto origin_exponent = other.value_exponent() + bias;
            // read url: https://baike.baidu.com/item/IEEE%20754/3869922?fr=aladdin
            // following code can make NaN number to max or min number
            constexpr auto max_exponent = (decltype(origin_exponent)(1) << exponent) - 2;
            constexpr auto max_fraction = (decltype(part_fraction)(1) << fraction) - 1;
            if (origin_exponent < decltype(origin_exponent)(0)) {
                origin_exponent = 0;
                part_fraction = 1;  // smallest float
            } else if (origin_exponent > max_exponent) {
                origin_exponent = max_exponent;
                part_fraction = max_fraction;
            }
            auto part_exponent = (origin_exponent << fraction) & this->mask_exponent().code;
            auto bits = part_sign | part_exponent | part_fraction;
            this->m_bits.code = typename bitset::type(bits);
        }

        bool iszero() const {
            return (this->m_bits.code & ~this->mask_sign().code) == 0;
        }

        inline uint32_t __f2i(float f) {
            union {
                uint32_t i;
                float f;
            } temp;
            temp.f = f;
            return temp.i;
        }

        inline uint64_t __f2i(double f) {
            union {
                uint64_t i;
                double f;
            } temp;
            temp.f = f;
            return temp.i;
        }

        ieee754_float(float f)
                : self(ieee754_float<32, 1, 8, 23>(ts::bitset_code<32>(__f2i(f)))) {
        }

        ieee754_float(double f)
                : self(ieee754_float<64, 1, 11, 52>(ts::bitset_code<64>(__f2i(f)))) {
        }

        ieee754_float(char i) : ieee754_float(float(i)) {}

        ieee754_float(char16_t i) : ieee754_float(float(i)) {}

        ieee754_float(char32_t i) : ieee754_float(float(i)) {}

        ieee754_float(int8_t i) : ieee754_float(float(i)) {}

        ieee754_float(uint8_t i) : ieee754_float(float(i)) {}

        ieee754_float(int16_t i) : ieee754_float(float(i)) {}

        ieee754_float(uint16_t i) : ieee754_float(float(i)) {}

        ieee754_float(int32_t i) : ieee754_float(double(i)) {}

        ieee754_float(uint32_t i) : ieee754_float(double(i)) {}

        ieee754_float(int64_t i) : ieee754_float(double(i)) {}

        ieee754_float(uint64_t i) : ieee754_float(double(i)) {}

        const typename bitset::type &code() const {
            return m_bits.code;
        }

        operator float() const {
            ieee754_float<32, 1, 8, 23> converted(*this);
            return *reinterpret_cast<const float *>(&converted.code());
        }

        operator double() const {
            ieee754_float<64, 1, 11, 52> converted(*this);
            return *reinterpret_cast<const double *>(&converted.code());
        }

#define DECLARE_OPERATOR(type) operator type() const { return type(double(*this)); }
        DECLARE_OPERATOR(char)

        DECLARE_OPERATOR(char16_t)

        DECLARE_OPERATOR(char32_t)

        DECLARE_OPERATOR(int8_t)

        DECLARE_OPERATOR(uint8_t)

        DECLARE_OPERATOR(int16_t)

        DECLARE_OPERATOR(uint16_t)

        DECLARE_OPERATOR(int32_t)

        DECLARE_OPERATOR(uint32_t)

        DECLARE_OPERATOR(int64_t)

        DECLARE_OPERATOR(uint64_t)

#undef DECLARE_OPERATOR

        friend self operator+(const self &lhs, const self &rhs) {
            return self(typename upper_float<_W>::type(lhs) + typename upper_float<_W>::type(rhs));
        }

        friend self operator-(const self &lhs, const self &rhs) {
            return self(typename upper_float<_W>::type(lhs) - typename upper_float<_W>::type(rhs));
        }

        friend self operator*(const self &lhs, const self &rhs) {
            return self(typename upper_float<_W>::type(lhs) * typename upper_float<_W>::type(rhs));
        }

        friend self operator/(const self &lhs, const self &rhs) {
            return self(typename upper_float<_W>::type(lhs) / typename upper_float<_W>::type(rhs));
        }

        friend self operator-(const self &x) {
            return self(-typename upper_float<_W>::type(x));
        }

        friend bool operator==(const self &lhs, const self &rhs) {
            if (lhs.iszero()) {
                return rhs.iszero();
            } else if (rhs.iszero()) {
                return false;
            }
            return lhs.code() == rhs.code();
        }

        friend bool operator!=(const self &lhs, const self &rhs) {
            return !operator==(lhs, rhs);
        }

        friend bool operator>=(const self &lhs, const self &rhs) {
            return lhs == rhs || (
                    ((lhs.code() | rhs.code()) & self::mask_sign().code) == 0 ?
                    lhs.code() > rhs.code() : lhs.code() < rhs.code()
            );
        }

        friend bool operator<=(const self &lhs, const self &rhs) {
            return lhs == rhs || (
                    ((lhs.code() | rhs.code()) & self::mask_sign().code) == 0 ?
                    lhs.code() < rhs.code() : lhs.code() > rhs.code()
            );
        }

        friend bool operator>(const self &lhs, const self &rhs) {
            return !operator<=(lhs, rhs);
        }

        friend bool operator<(const self &lhs, const self &rhs) {
            return !operator>=(lhs, rhs);
        }

        friend std::ostream &operator<<(std::ostream &out, const self &x) {
            return out << double(x);
        }

        friend std::istream &operator>>(std::istream &in, self &x) {
            double d;
            in >> d;
            x = d;
            return in;
        }

    private:
        bitset m_bits;

        uint64_t value_sign() const {
            return (this->m_bits.code & this->mask_sign().code) >> (W - 1);
        }

        int64_t value_exponent() const {
            return int64_t((this->m_bits.code & this->mask_exponent().code) >> fraction)
                   - int64_t(bias);
        }

        uint64_t value_fraction() const {
            return this->m_bits.code & this->mask_fraction().code;
        }

        static constexpr bitset mask_fraction() TS_NOEXCEPT {
            return bitset((typename bitset::type(0x01) << fraction) - 1);
        }

        static constexpr bitset mask_exponent() TS_NOEXCEPT {
            return bitset(((typename bitset::type(0x01) << exponent) - 1) << fraction);
        }

        static constexpr bitset mask_sign() TS_NOEXCEPT {
            return bitset(typename bitset::type(0x01) << (W - 1));
        }

    };

    template<>
    inline ieee754_float<32, 1, 8, 23>::ieee754_float(float f) {
        m_bits = bitset(__f2i(f));
    }

    template<>
    inline ieee754_float<64, 1, 11, 52>::ieee754_float(double f) {
        m_bits = bitset(__f2i(f));
    }

    using float16 = ieee754_float<16, 1, 5, 10>;
    using float32 = ieee754_float<32, 1, 8, 23>;
    using float64 = ieee754_float<64, 1, 11, 52>;


    template <> struct dtype<FLOAT16> { using declare = float16; };
    template <> struct dtypeid<float16> { static const DTYPE id = FLOAT16; };
}

extern template class ts::tensor_builder<ts::dtype<ts::FLOAT16>::declare>;


#endif //TENSORSTACK_CORE_IEEE754_FLOAT_H
