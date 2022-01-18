//
// Created by lby on 2018/1/23.
//

#include "orz/tools/progress_bar.h"
#include <sstream>
#include <iomanip>

namespace orz {

    static bool append_string(std::string &base, const std::string &add, size_t limit) {
        if (base.length() + add.length() > limit) return false;
        base.insert(base.end(), add.begin(), add.end());
        return true;
    }

    static void output_string(std::ostream &out) { (decltype(output_string(out))()); }

    template<typename T, typename... Args>
    static void output_string(std::ostream &out, T &&t, Args &&... args) {
        output_string(out << std::forward<T>(t), std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline const std::string concat_string(std::ostringstream &oss, Args &&... args) {
        oss.str("");
        output_string(oss, std::forward<Args>(args)...);
        return oss.str();
    }

    std::string to_string(microseconds us, size_t limit) {
        auto count = us.count();
        count /= 1000L;    // ms
        auto day = count / (24L * 60L * 60L * 1000L);
        count %= (24L * 60L * 60L * 1000L);
        auto hour = count / (60L * 60L * 1000L);
        count %= (60L * 60L * 1000L);
        auto minute = count / (60L * 1000L);
        count %= (60L * 1000L);
        auto second = count / (1000L);
        count %= (1000L);
        auto ms = count;

        std::string format;
        std::ostringstream oss;
        if (day && !append_string(format, concat_string(oss, day, 'd'), limit)) return format;
        if (hour && !append_string(format, concat_string(oss, hour, 'h'), limit)) return format;
        if (minute && !append_string(format, concat_string(oss, minute, 'm'), limit)) return format;
        if (second && !append_string(format, concat_string(oss, second, 's'), limit)) return format;
        if (!day && !hour && !minute && ms && !append_string(format, concat_string(oss, ms, "ms"), limit))
            return format;

        return format;
    }

    int progress_bar::next() {
        return next(m_step);
    }

    int progress_bar::next(int step) {
        m_value += step;
        if (m_value >= m_max && m_autostop) {
            self::stop();
            m_value = m_max;
        }

        if (stat() == RUNNING) {
            sample();
        }

        return m_value;
    }

    microseconds progress_bar::used_time() const {
        switch (m_stat) {
            default:
                return microseconds(0);
            case WAITING:
                return microseconds(0);
            case RUNNING:
                return duration_cast<microseconds>(system_clock::now() - m_start_time_point) - m_paused_duration;
            case PAUSED:
                return duration_cast<microseconds>(m_pause_time_point - m_start_time_point) - m_paused_duration;
            case STOPPED:
                return duration_cast<microseconds>(m_stop_time_point - m_start_time_point) - m_paused_duration;
        }
    }

    microseconds progress_bar::left_time() const {
        if (m_vpus == 0) {
            auto used_time = self::used_time();
            if (used_time.count() == 0) return microseconds(0);
            auto proessed_count = m_value - m_min;
            auto left_count = m_max - m_value;
            if (proessed_count == 0) return microseconds(0);
            return used_time * left_count / proessed_count;
        }

        auto left_count = m_max - m_value;
        return microseconds(int64_t(left_count / m_vpus));
    }

    progress_bar::progress_bar(int min, int max, int value)
            : m_min(min), m_max(max), m_value(value), m_paused_duration(0) {
        m_last_show_time_point = system_clock::now() - std::chrono::seconds(3600);
    }

    progress_bar::progress_bar(int min, int max) : progress_bar(min, max, min) {}

    progress_bar::progress_bar(int max) : progress_bar(0, max, 0) {}

    progress_bar::progress_bar() : progress_bar(0, 100, 0) {}

    void progress_bar::start() {
        switch (m_stat) {
            default:
                m_start_time_point = system_clock::now();
                reset();
                break;
            case WAITING:
                m_start_time_point = system_clock::now();
                m_paused_duration = microseconds(0);
                reset();
                break;
            case RUNNING:
                break;
            case PAUSED:
                m_paused_duration += duration_cast<microseconds>(system_clock::now() - m_pause_time_point);
                reset();
                break;
            case STOPPED:
                m_start_time_point = system_clock::now();
                m_paused_duration = microseconds(0);
                reset();
                break;
        }
        m_stat = RUNNING;
    }

    void progress_bar::stop() {
        switch (m_stat) {
            default:
                m_stop_time_point = system_clock::now();
                break;
            case WAITING:
                m_start_time_point = system_clock::now();
                m_stop_time_point = m_start_time_point;
                break;
            case RUNNING:
                m_stop_time_point = system_clock::now();
                break;
            case PAUSED:
                m_paused_duration += duration_cast<microseconds>(system_clock::now() - m_pause_time_point);
                m_stop_time_point = system_clock::now();
                break;
            case STOPPED:
                break;
        }
        m_stat = STOPPED;
    }

    void progress_bar::pause() {
        switch (m_stat) {
            default:
                m_pause_time_point = system_clock::now();
                break;
            case WAITING:
                m_start_time_point = system_clock::now();
                m_pause_time_point = m_start_time_point;
                break;
            case RUNNING:
                m_pause_time_point = system_clock::now();
                break;
            case PAUSED:
                break;
            case STOPPED:
                break;
        }
        m_stat = PAUSED;
    }

    void progress_bar::autostop(bool flag) {
        m_autostop = flag;
    }

    bool progress_bar::autostop() const {
        return m_autostop;
    }

    int progress_bar::value() const {
        return m_value;
    }

    int progress_bar::max() const {
        return m_max;
    }

    int progress_bar::min() const {
        return m_min;
    }

    void progress_bar::set_value(int value) {
        m_value = value;
    }

    void progress_bar::set_min(int min) {
        m_min = min;
    }

    void progress_bar::set_max(int max) {
        m_max = max;
    }

    progress_bar::status progress_bar::stat() const {
        return m_stat;
    }

    static const char running_status[] = {'-', '\\', '|', '/'};
    static const auto running_status_num = sizeof(running_status) / sizeof(running_status[0]);

    std::ostream &progress_bar::show(std::ostream &out) const {
        std::ostringstream oss;

        switch (m_stat) {
            case WAITING:
                oss << "[~]";
                break;
            case RUNNING:
                oss << '[' << running_status[m_show_count % running_status_num] << ']';
                m_show_count++;
                m_show_count %= running_status_num;
                break;
            case PAUSED:
                oss << "[=]";
                break;
            case STOPPED:
                oss << "[*]";
                break;
        }

        int ps = percent();
        int processed = ps / 2;
        int left = (100 - ps) / 2;
        oss << '[';
        for (int i = 0; i < processed; ++i) oss << '>';
        if (ps % 2) oss << '=';
        for (int i = 0; i < left; ++i) oss << '-';
        oss << ']';

        if (ps == 100) oss << "[--%]";
        else oss << '[' << std::setw(2) << ps << "%]";

        oss << '[';
        oss << std::setw(8) << to_string(used_time(), 8);
        oss << '/';
        oss << std::setw(8) << to_string(left_time(), 8);
        oss << ']';

        oss << '\r';
        out << oss.str() << std::flush;

        return out;
    }

    int progress_bar::percent() const {
        auto fpercent = float(m_value - m_min) / (m_max - m_min) * 100;
        auto ipercent = static_cast<int>(fpercent);
        return ipercent;
    }

    void progress_bar::reset() {
        m_sample_value = value();
        m_sample_time_point = system_clock::now();
        m_vpus = 0;
    }

    void progress_bar::sample() {
        // 60 count or 1 secend simple rate
        auto now_value = value();
        auto now_time_point = system_clock::now();
        auto sample_time_duration = duration_cast<microseconds>(now_time_point - m_sample_time_point);
        auto sample_value_duration = now_value - m_sample_value;
        if (sample_time_duration > seconds(10) && sample_value_duration > 0) {
            m_vpus = double(sample_value_duration) / sample_time_duration.count();
            m_sample_value = now_value;
            m_sample_time_point = now_time_point;
        }
    }

    std::ostream &progress_bar::wait_show(int ms, std::ostream &out) const {
        auto now_time_point = system_clock::now();
        auto wait_duration = duration_cast<milliseconds>(now_time_point - m_last_show_time_point);
        if (wait_duration.count() >= ms) {
            m_last_show_time_point = now_time_point;
            return show(out);
        }
        return out;
    }
}
