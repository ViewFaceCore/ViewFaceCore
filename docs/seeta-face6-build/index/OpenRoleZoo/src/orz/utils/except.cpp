//
// Created by Lby on 2017/6/6.
//

#include "orz/utils/except.h"

namespace orz {
    Exception::Exception(const std::string &message)
            : m_message(message) {}

    const char *Exception::what() const ORZ_NOEXCEPT {
        return m_message.c_str();
    }
}