//
// Created by kier on 2019/3/27.
//

#ifndef TENSORSTACK_BOARD_TIME_LOG_H
#define TENSORSTACK_BOARD_TIME_LOG_H

#include <chrono>
#include <string>

#include "utils/log.h"

namespace ts {
    class TimeLog {
    public:
        using self = TimeLog;

        using microseconds = std::chrono::microseconds;
        using system_clock = std::chrono::system_clock;
        using time_point = decltype(system_clock::now());

        explicit TimeLog(ts::LogLevel level, const std::string &header = "", float *time= nullptr) :
                m_duration(0) {
            m_level = level;
            m_header = header;
            m_time = time;
            m_start = system_clock::now();
        }

        explicit TimeLog(ts::LogLevel level, float *time) : self(level, "", time) {}

        void denominator(float d) { m_denominator = d; }
        float denominator() const { return m_denominator; }

        ~TimeLog() {
            m_end = system_clock::now();
            m_duration = std::chrono::duration_cast<microseconds>(m_end - m_start);

            float time = m_duration.count() / 1000.0f / m_denominator;
            if (m_time) *m_time = time;

            std::ostringstream oss;
            ts::LogStream(m_level) << m_header << time << "ms";
        }

        TimeLog(const self &) = delete;
        self &operator=(const self &) = delete;

    private:
        ts::LogLevel m_level;
        std::string m_header;
        microseconds m_duration;
        time_point m_start;
        time_point m_end;
        float *m_time = nullptr;
        float m_denominator = 1.0f;
    };
}

#endif //TENSORSTACK_BOARD_TIME_LOG_H
