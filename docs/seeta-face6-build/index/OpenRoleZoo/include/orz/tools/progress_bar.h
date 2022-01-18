//
// Created by lby on 2018/1/23.
//

#ifndef ORZ_TOOLS_PROGRESS_BAR_H
#define ORZ_TOOLS_PROGRESS_BAR_H

#include <chrono>
#include <string>
#include <iostream>

namespace orz {
    using system_clock = std::chrono::system_clock;
    using time_point = decltype(system_clock::now());
    using microseconds = std::chrono::microseconds;
    using milliseconds = std::chrono::milliseconds;
    using seconds = std::chrono::seconds;
    using std::chrono::duration_cast;

    std::string to_string(microseconds us, size_t limit = 8);

    class progress_bar {
    public:
        using self = progress_bar;

        enum status {
            WAITING,
            RUNNING,
            PAUSED,
            STOPPED,
        };

        progress_bar(int min, int max, int value);

        progress_bar(int min, int max);

        progress_bar(int max);

        progress_bar();

        status stat() const;

        void start();

        void stop();

        void pause();

        void autostop(bool flag);

        bool autostop() const;

        int value() const;

        int max() const;

        int min() const;

        void set_value(int value);

        void set_min(int min);

        void set_max(int max);

        int next();

        int next(int step);

        microseconds used_time() const;

        microseconds left_time() const;

        int percent() const;

        std::ostream &show(std::ostream &out) const;

        std::ostream &wait_show(int ms, std::ostream &out) const;

    private:
        // reset sample
        void reset();

        // sample value and time point, calculate speed
        void sample();

        int m_min;
        int m_max;
        int m_value;
        int m_step = 1;

        bool m_autostop = true;

        status m_stat = WAITING;

        time_point m_start_time_point;
        time_point m_stop_time_point;
        time_point m_pause_time_point;
        microseconds m_paused_duration;

        mutable int m_show_count = 0;

        int m_sample_value;
        time_point m_sample_time_point;
        double m_vpus; // values per microseconds

        mutable time_point m_last_show_time_point;
    };
}



#endif //ORZ_TOOLS_PROGRESS_BAR_H
