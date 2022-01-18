//
// Created by Lby on 2017/6/6.
//

#ifndef TENSORSTACK_UTILS_EXCEPT_H
#define TENSORSTACK_UTILS_EXCEPT_H

#include <exception>
#include <string>

#include "platform.h"

#if TS_PLATFORM_CC_MSVC
#define TS_NOEXCEPT
#else
#define TS_NOEXCEPT noexcept
#endif

#include "api.h"

namespace ts {
    class TS_DEBUG_API Exception : public std::exception {
    public:
        Exception();
        explicit Exception(const std::string &message);

        const char *what() const TS_NOEXCEPT override;

    private:
        std::string m_message;
    };

    class TS_DEBUG_API NullPointerException : public Exception {
    public:
        NullPointerException() : Exception() {}
        explicit NullPointerException(const std::string &message) : Exception(message) {}
    };
}

#endif //TENSORSTACK_UTILS_EXCEPT_H
