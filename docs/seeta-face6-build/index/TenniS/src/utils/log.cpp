//
// Created by kier on 2018/11/5.
//

#include "utils/log.h"
#include "utils/static.h"

#include <atomic>

namespace ts {
    static std::atomic<LogLevel> InnerGlobalLogLevel;

    LogLevel GlobalLogLevel(LogLevel level) {
        LogLevel pre_level = InnerGlobalLogLevel;
        InnerGlobalLogLevel = level;
        return pre_level;
    }

    LogLevel GlobalLogLevel() {
        return InnerGlobalLogLevel;
    }
}

TS_STATIC_ACTION((ts::LogLevel(*)(ts::LogLevel))ts::GlobalLogLevel, ts::LOG_INFO)
