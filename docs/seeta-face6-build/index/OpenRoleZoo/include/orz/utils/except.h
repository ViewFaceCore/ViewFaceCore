//
// Created by Lby on 2017/6/6.
//

#ifndef ORZ_UTILS_EXCEPT_H
#define ORZ_UTILS_EXCEPT_H

#include <exception>
#include <string>

#include "platform.h"

#if ORZ_PLATFORM_CC_MSVC
#define ORZ_NOEXCEPT
#else
#define ORZ_NOEXCEPT noexcept
#endif

namespace orz {
    class Exception : public std::exception {
    public:
        Exception(const std::string &message);

        virtual const char *what() const ORZ_NOEXCEPT override;

    private:
        std::string m_message;
    };
}

#endif //ORZ_UTILS_EXCEPT_H
