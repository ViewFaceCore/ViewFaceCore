//
// Created by lby on 2018/3/12.
//

#ifndef TENSORSTACK_UTILS_STATIC_H
#define TENSORSTACK_UTILS_STATIC_H

#include <utility>
#include <functional>

#include "utils/except.h"

namespace ts {
    /**
     * StaticAction: for supporting static initialization
     */
    class StaticAction {
    public:
        template <typename FUNC, typename... Args>
        explicit StaticAction(FUNC func, Args&&... args) TS_NOEXCEPT {
            std::bind(func, std::forward<Args>(args)...)();
        }
    };
}

#define _ts_concat_name_core(x,y) (x##y)

#define _ts_concat_name(x, y) _ts_concat_name_core(x,y)

/**
 * generate an serial name by line
 */
#define ts_serial_name(x) _ts_concat_name(x, __LINE__)

/**
 * Static action
 */
#define TS_STATIC_ACTION(func, ...) \
    namespace \
    { \
         ts::StaticAction ts_serial_name(_ts_static_action_)(func, ## __VA_ARGS__); \
    }

#endif //TENSORSTACK_UTILS_STATIC_H
