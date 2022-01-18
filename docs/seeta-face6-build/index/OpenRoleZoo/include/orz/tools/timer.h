//
// Created by seeta on 2018/8/3.
//

#ifndef ORZ_TOOLS_TIMER_H
#define ORZ_TOOLS_TIMER_H

#include <chrono>

namespace orz {

    /**
     * using microseconds for timer
     */
    class Timer {
    public:
        using duration = std::chrono::microseconds;
        using time_point = std::chrono::system_clock::time_point;
        using system_clock = std::chrono::system_clock;

        Timer() {
            m_start = system_clock::now();
        }

        duration reset() {
            auto now = system_clock::now();
            auto count = std::chrono::duration_cast<duration>(now - m_start);
            m_start = now;
            return count;
        }

        duration count() const {
            auto now = system_clock::now();
            return std::chrono::duration_cast<duration>(now - m_start);
        }

        time_point start() const {
            return m_start;
        }

        time_point now() const {
            return system_clock::now();
        }

    private:
        time_point m_start;
    };
}

#endif //ORZ_TOOLS_TIMER_H
