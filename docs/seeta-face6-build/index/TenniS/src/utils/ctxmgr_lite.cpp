//
// Created by kier on 2019-04-14.
//

#include "utils/ctxmgr_lite.h"
#include "utils/platform.h"

#include <sstream>

namespace ts {
    static inline std::string no_lite_build_message(const std::thread::id &id) {
        std::ostringstream oss;
        oss << "Empty context in thread: " << id;
        return oss.str();
    }

    static inline std::string no_lite_build_message(const std::string &name, const std::thread::id &id) {
        std::ostringstream oss;
        oss << "Empty context:<" << classname(name) << "> in thread: " << id;
        return oss.str();
    }

    NoLiteContextException::NoLiteContextException()
            : NoLiteContextException(std::this_thread::get_id()) {
    }

    NoLiteContextException::NoLiteContextException(const std::thread::id &id)
            : Exception(no_lite_build_message(id)), m_thread_id(id) {
    }

    NoLiteContextException::NoLiteContextException(const std::string &name)
            : NoLiteContextException(name, std::this_thread::get_id()) {
    }

    NoLiteContextException::NoLiteContextException(const std::string &name, const std::thread::id &id)
            : Exception(no_lite_build_message(name, id)), m_thread_id(id) {
    }

#if TS_PLATFORM_CC_GCC
#include <cxxabi.h>
    static ::std::string classname_gcc(const ::std::string &name) {
        size_t size = 0;
        int status = 0;
        char *demangled = abi::__cxa_demangle(name.c_str(), nullptr, &size, &status);
        if (demangled != nullptr) {
            ::std::string parsed = demangled;
            ::std::free(demangled);
            return parsed;
        } else {
            return name;
        }
    }
#endif

    ::std::string classname(const ::std::string &name) {
#if TS_PLATFORM_CC_MSVC
        return name;
#elif TS_PLATFORM_CC_MINGW
        return name;
#elif TS_PLATFORM_CC_GCC
        return classname_gcc(name);
#else
        return name;
#endif
    }
}

