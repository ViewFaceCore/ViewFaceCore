//
// Created by lby on 2018/3/11.
//

#include "core/device.h"

namespace ts {
    bool operator<(const Device &lhs, const Device &rhs) {
        return lhs.type() < rhs.type() ? true : (lhs.type() == rhs.type() && lhs.id() < rhs.id());
    }

    bool operator>(const Device &lhs, const Device &rhs) {
        return lhs.type() > rhs.type() ? true : (lhs.type() == rhs.type() && lhs.id() > rhs.id());
    }

    bool operator<=(const Device &lhs, const Device &rhs) {
        return lhs.type() > rhs.type() ? false : (lhs.type() != rhs.type() || lhs.id() <= rhs.id());
    }

    bool operator>=(const Device &lhs, const Device &rhs) {
        return lhs.type() < rhs.type() ? false : (lhs.type() != rhs.type() || lhs.id() >= rhs.id());
    }

    bool operator==(const Device &lhs, const Device &rhs) {
        return lhs.type() == rhs.type() && lhs.id() == rhs.id();
    }

    bool operator!=(const Device &lhs, const Device &rhs) {
        return lhs.type() != rhs.type() || lhs.id() != rhs.id();
    }

    DeviceMismatchException::DeviceMismatchException(const Device &needed, const Device &given)
            : Exception(DeviceMismatchMessage(needed, given)), m_needed(needed), m_given(given) {
    }

    std::string DeviceMismatchException::DeviceMismatchMessage(const Device &needed, const Device &given) {
        std::ostringstream oss;
        oss << "Given device " << given
            << ", " << needed << " expected.";
        return oss.str();
    }
}
