//
// Created by kier on 2019/9/7.
//

#ifndef TENSORSTACK_THIRD_DRAGON_LOGGING_H
#define TENSORSTACK_THIRD_DRAGON_LOGGING_H

#include "utils/log.h"

namespace ts {
    namespace dragon {
#define CHECK_EQ TS_CHECK_EQ
#define CHECK_GT TS_CHECK_GT
#define LOG TS_LOG

        static const LogLevel FATAL = LOG_FATAL;

#define CPU_FP16_NOT_SUPPORTED TS_LOG_ERROR << "CPU float16 not supported." << ts::eject

    }
}
#endif //TENSORSTACK_THIRD_DRAGON_LOGGING_H
