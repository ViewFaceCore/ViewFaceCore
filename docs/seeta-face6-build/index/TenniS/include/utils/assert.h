//
// Created by kier on 2018/11/6.
//

#ifndef TENSORSTACK_UTILS_ASSERT_H
#define TENSORSTACK_UTILS_ASSERT_H

#include "log.h"

#include <cfloat>

namespace ts {
    template <typename T>
    class __near_zero {
    public:
        using check_value = T;
        static bool check(check_value value) { return value == 0; }
    };

    template <>
    class __near_zero<double> {
    public:
        using check_value = double;
        static bool check(check_value value) {
            return (value > 0.0 ? value - 0.0 : 0.0 - value) < DBL_EPSILON;
        }
    };

    template <>
    class __near_zero<float> {
    public:
        using check_value = float;
        static bool check(check_value value) {
            return (value > 0.0f ? value - 0.0f : 0.0f - value) < FLT_EPSILON;
        }
    };

    template <typename T>
    inline bool near_zero(T value) {
        return __near_zero<T>::check(value);
    }
}

#define TS_ASSERT(condition) TS_LOG((condition) ? ts::LOG_NONE : ts::LOG_FATAL)("Assertion failed: (")(#condition)("). ")
#define TS_CHECK(condition) TS_LOG((condition) ? ts::LOG_NONE : ts::LOG_ERROR)("Check failed: (")(#condition)("). ")

#define TS_CHECK_EQ(lhs, rhs) TS_CHECK((lhs) == (rhs))
#define TS_CHECK_NQ(lhs, rhs) TS_CHECK((lhs) != (rhs))

#define TS_CHECK_LE(lhs, rhs) TS_CHECK((lhs) <= (rhs))
#define TS_CHECK_GE(lhs, rhs) TS_CHECK((lhs) >= (rhs))

#define TS_CHECK_LT(lhs, rhs) TS_CHECK((lhs) < (rhs))
#define TS_CHECK_GT(lhs, rhs) TS_CHECK((lhs) > (rhs))

#define TS_AUTO_ASSERT(condition) (TS_LOG((condition) ? ts::LOG_NONE : ts::LOG_FATAL)("Assertion failed: (")(#condition)(").") << ts::fatal)
#define TS_AUTO_CHECK(condition) (TS_LOG((condition) ? ts::LOG_NONE : ts::LOG_ERROR)("Check failed: (")(#condition)(").") << ts::eject)

#define TS_AUTO_CHECK_EQ(lhs, rhs) TS_AUTO_CHECK((lhs) == (rhs))
#define TS_AUTO_CHECK_NQ(lhs, rhs) TS_AUTO_CHECK((lhs) != (rhs))

#define TS_AUTO_CHECK_LE(lhs, rhs) TS_AUTO_CHECK((lhs) <= (rhs))
#define TS_AUTO_CHECK_GE(lhs, rhs) TS_AUTO_CHECK((lhs) >= (rhs))

#define TS_AUTO_CHECK_LT(lhs, rhs) TS_AUTO_CHECK((lhs) < (rhs))
#define TS_AUTO_CHECK_GT(lhs, rhs) TS_AUTO_CHECK((lhs) > (rhs))

#endif //TENSORSTACK_UTILS_ASSERT_H
