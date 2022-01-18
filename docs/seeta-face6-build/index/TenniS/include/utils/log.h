//
// Created by kier on 2018/11/5.
//

#ifndef TENSORSTACK_UTILS_LOG_H
#define TENSORSTACK_UTILS_LOG_H



#include <iostream>
#include <sstream>
#include <cstring>

#include "except.h"
#include "box.h"
#include "platform.h"

#if TS_PLATFORM_OS_ANDROID
#include "android/log.h"
#define ANDROID_LOG_TAG "TenniS"
#define ANDROID_LOG(LEVEL, ...) __android_log_print(LEVEL,          ANDROID_LOG_TAG, __VA_ARGS__)
#define ANDROID_LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE,  ANDROID_LOG_TAG, __VA_ARGS__)
#define ANDROID_LOGI(...) __android_log_print(ANDROID_LOG_INFO,     ANDROID_LOG_TAG, __VA_ARGS__)
#define ANDROID_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,    ANDROID_LOG_TAG, __VA_ARGS__)
#define ANDROID_LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    ANDROID_LOG_TAG, __VA_ARGS__)
#endif

namespace ts {

    class TS_DEBUG_API EjectionException : public Exception {
    public:
        EjectionException() : Exception() {}
        explicit EjectionException(const std::string &message) : Exception(message) {}
    };

    enum LogLevel {
        LOG_NONE = 0,
        LOG_DEBUG = 1,
        LOG_STATUS = 2,
        LOG_INFO = 3,
        LOG_ERROR = 4,
        LOG_FATAL = 5,
    };

#if TS_PLATFORM_OS_ANDROID
    inline android_LogPriority __android_log_level(LogLevel level) {
        switch (level) {
            default: return ANDROID_LOG_UNKNOWN;
            case LOG_NONE: return ANDROID_LOG_VERBOSE;
            case LOG_DEBUG: return ANDROID_LOG_DEBUG;
            case LOG_STATUS: return ANDROID_LOG_INFO;
            case LOG_INFO: return ANDROID_LOG_INFO;
            case LOG_ERROR: return ANDROID_LOG_ERROR;
            case LOG_FATAL: return ANDROID_LOG_FATAL;
        }
    }
#endif

    inline std::string LogString(LogLevel level) {
        switch (level) {
            default:
                return "[Unknown]";
            case LOG_NONE:
                return "";
            case LOG_DEBUG:
                return "[DEBUG]";
            case LOG_STATUS:
                return "[STATUS]";
            case LOG_INFO:
                return "[INFO]";
            case LOG_ERROR:
                return "[ERROR]";
            case LOG_FATAL:
                return "[FATAL]";
        }
    }

    TS_DEBUG_API LogLevel GlobalLogLevel(LogLevel level);

    TS_DEBUG_API LogLevel GlobalLogLevel();

    class TS_DEBUG_API LogStream {
    public:
        using self = LogStream;

        explicit LogStream(LogLevel level, std::ostream &log = std::cout)
                : m_level(level), m_log(log) {
        }

        LogStream(const self &other) = delete;

        self &operator=(const self &other) = delete;

        ~LogStream() {
            flush();
        }

        const std::string message() const {
            return m_buffer.str();
        }

        template<typename T>
        self &operator()(T &&message) {
            if (m_level == LOG_NONE) return *this;
            if (m_level >= GlobalLogLevel()) {
                m_buffer << std::forward<T>(message);
            }
            return *this;
        }

        template<typename T>
        self &operator<<(T &&message) {
            return operator()(std::forward<T>(message));
        }

        using Method = self &(self &);

        self &operator<<(Method method) {
            if (m_level == LOG_NONE) return *this;
            if (m_level >= GlobalLogLevel()) {
                return method(*this);
            }
            return *this;
        }

        void flush() {
            if (m_level == LOG_NONE) return;
            if (m_level >= GlobalLogLevel()) {
                auto msg = m_buffer.str();
                m_buffer.str("");
                m_buffer << LogString(m_level) << ": " << msg << std::endl;
                m_log << m_buffer.str();
#if TS_PLATFORM_OS_ANDROID
                ANDROID_LOG(__android_log_level(m_level), "%s", msg.c_str());
#endif
            }
#if TS_PLATFORM_OS_ANDROID
            else {
                auto msg = m_buffer.str();
                ANDROID_LOG(__android_log_level(m_level), "%s", msg.c_str());
            }
#endif
            m_level = LOG_NONE;
            m_buffer.str("");
            m_log.flush();
        }

        LogLevel level() const { return m_level; }

    private:
        LogLevel m_level;
        std::ostringstream m_buffer;
        std::ostream &m_log;
    };

    inline LogStream &fatal(LogStream &log) {
        const auto msg = log.message();
        log.flush();
        std::exit(-1);
    }

    inline LogStream &eject(LogStream &log) {
        const auto msg = log.message();
        log.flush();
        throw EjectionException(msg);
    }
}

#ifdef TS_SOLUTION_DIR
#define TS_LOCAL_FILE ( \
    std::strlen(TS_SOLUTION_DIR) + 1 < std::strlen(__FILE__) \
    ? ((const char *)(__FILE__) + std::strlen(TS_SOLUTION_DIR) + 1) \
    : ((const char *)(__FILE__)) \
    )
#else
#define TS_LOCAL_FILE (ts::Split(__FILE__, R"(/\)").back())
#endif

#define TS_LOG(level) (ts::LogStream(level))("[")(TS_LOCAL_FILE)(":")(__LINE__)("]: ")
#define TS_TIME(level) (ts::LogStream(level))("[")(ts::now_time())("]: ")
#define TS_LOG_TIME(level) (ts::LogStream(level))("[")(TS_LOCAL_FILE)(":")(__LINE__)("][")(ts::now_time())("]: ")

#define TS_LOG_DEBUG TS_LOG(ts::LOG_DEBUG)
#define TS_LOG_STATUS TS_LOG(ts::LOG_STATUS)
#define TS_LOG_INFO TS_LOG(ts::LOG_INFO)
#define TS_LOG_ERROR TS_LOG(ts::LOG_ERROR)
#define TS_LOG_FATAL TS_LOG(ts::LOG_FATAL)

#endif //TENSORSTACK_LOG_H
