//
// Created by Lby on 2017/6/6.
//

#include "utils/except.h"

namespace ts {
    Exception::Exception(const std::string &message)
            : m_message(message) {}

    const char *Exception::what() const TS_NOEXCEPT {
        return m_message.c_str();
    }

    Exception::Exception()
            : m_message("Unknown exception.") {
    }
}