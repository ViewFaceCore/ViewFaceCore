//
// Created by kier on 2018/7/19.
//

#ifndef TENSORSTACK_KERNELS_COMMON_MATH_H
#define TENSORSTACK_KERNELS_COMMON_MATH_H

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <math.h>
#include <cstdint>

namespace ts {
    template <typename T>
    inline bool near(T value1, T value2) {
        return (value1 - value2 == 0);
    }

    template<>
    inline bool near<double>(double value1, double value2) {
        return (value1 > value2 ? value1 - value2 : value2 - value1) < DBL_EPSILON;
    }

    template<>
    inline bool near<float>(float value1, float value2) {
        return (value1 > value2 ? value1 - value2 : value2 - value1) < FLT_EPSILON;
    }

    template <typename T>
    inline T abs(T value) {
        return T(std::abs(value));
    }

    template <>
    inline uint8_t abs(uint8_t value) {
        return value;
    }

    template <>
    inline uint16_t abs(uint16_t value) {
        return value;
    }

    template <>
    inline uint32_t abs(uint32_t value) {
        return value;
    }

    template <>
    inline uint64_t abs(uint64_t value) {
        return value;
    }

    template <>
    inline float abs(float value) {
        using namespace std;
        return fabsf(value);
    }

    template <>
    inline double abs(double value) {
        return std::fabs(value);
    }

    template <typename T>
    inline T round_up(T i, T factor) {
        return (i + factor - 1) / factor * factor;
    }

}


#endif //TENSORSTACK_KERNELS_COMMON_MATH_H
