//
// Created by kier on 2019/1/29.
//

#ifndef TENSORSTACK_BOARD_PROFILER_H
#define TENSORSTACK_BOARD_PROFILER_H

#include <unordered_map>
#include <string>
#include <cstdint>
#include <functional>

#include "utils/except.h"
#include "utils/ctxmgr_lite.h"

#include "board.h"

#include <utils/api.h>

namespace ts {
    class TS_DEBUG_API Later {
    public:
        using self = Later;

        Later() = default;

        template<typename FUNC>
        Later(FUNC func)
                : m_later(func) {
        }

        template<typename FUNC, typename ...Args>
        Later(FUNC func, Args &&...args)
                : m_later(std::bind(func, std::forward<Args>(args)...)) {
        }

        ~Later() {
            if (m_later) m_later();
        }

        Later(const self &) = delete;

        Later &operator=(const self &) = delete;

        Later(self &&other) TS_NOEXCEPT {
            std::swap(this->m_later, other.m_later);
        }

        Later &operator=(self &&other) TS_NOEXCEPT {
            std::swap(this->m_later, other.m_later);
            return *this;
        }

    private:
        std::function<void(void)> m_later;
    };

    class TS_DEBUG_API Profiler : public SetupContext<Profiler> {
    public:
        void clean() {
            this->m_serial.clear();
        }

        Later timer(const std::string &name);

        /**
         * query or add serial in system
         * @param name
         * @return
         */
        int32_t serial_of(const std::string &name);

        Board<float> &board();

        const Board<float> &board() const;

        void log(std::ostream &out) const;
    private:
        Board<float> m_board;
        std::unordered_map<std::string, int32_t> m_serial;
    };

    TS_DEBUG_API bool profiler_on();
    /**
     *
     * @param name
     * @return
     * @note name must include format string of integer, such as "%04d".
     */
    TS_DEBUG_API Later profiler_serial_timer(const std::string &name);

    /**
     *
     * @param name
     * @return just write name
     */
    TS_DEBUG_API Later profiler_timer(const std::string &name);
}


#endif //TENSORSTACK_BOARD_PROFILER_H
