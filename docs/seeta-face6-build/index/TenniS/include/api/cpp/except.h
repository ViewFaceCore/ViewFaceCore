//
// Created by kier on 2019/3/16.
//

#ifndef TENNIS_API_CPP_EXCEPT_H
#define TENNIS_API_CPP_EXCEPT_H

#include "../common.h"
#include <exception>
#include <string>

#if defined(_MSC_VER) && _MSC_VER < 1900 // lower then VS2015
#define TS_API_NOEXCEPT
#else
#define TS_API_NOEXCEPT noexcept
#endif

namespace ts {
    namespace api {
        class Exception : public std::exception {
        public:
            using self = Exception;

            Exception() : Exception(ts_last_error_message()) {}

            explicit Exception(const std::string &message) : m_message(message) {}

            const char *what() const TS_API_NOEXCEPT override { return m_message.c_str(); }

        private:
            std::string m_message;
        };
    }
}

#define TS_API_AUTO_CHECK(condition) \
    if (!(condition)) { throw ts::api::Exception(); }

#endif //TENNIS_API_CPP_EXCEPT_H
