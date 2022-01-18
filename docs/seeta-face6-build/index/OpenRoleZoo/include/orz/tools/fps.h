//
// Created by lby on 2018/6/10.
//

#ifndef ORZ_TOOLS_FPS_H
#define ORZ_TOOLS_FPS_H

#include <chrono>
#include <deque>
#include <cmath>

namespace orz {
    class FPS {
    public:
        using system_clock = std::chrono::system_clock;
        using time_point = decltype(system_clock::now());
        using microseconds = std::chrono::microseconds;
        using milliseconds = std::chrono::milliseconds;
        using seconds = std::chrono::seconds;

        explicit FPS(seconds range = seconds(1)) : m_range(range) {}

        void rewind() {
            this->m_every_time_point.clear();
        }

        FPS &tick() {
            auto now = system_clock::now();
            double now_fps = 0;
            if (this->m_every_time_point.empty()) {
                this->m_every_time_point.push_back(now);
            } else {
                this->m_every_time_point.push_back(now);

                now_fps = m_range.count() * double(1000000)
                          / std::chrono::duration_cast<microseconds>(
                        this->m_every_time_point.back() - this->m_every_time_point.front()).count()
                          * (this->m_every_time_point.size() - 1);
                while (this->m_every_time_point.size() > 1 &&
                       (now - this->m_every_time_point.front()) > this->m_range)  // seconds over m_range will be ignored
                    this->m_every_time_point.pop_front();
            }
            this->m_fps = static_cast<decltype(this->m_fps)>(now_fps);
            return *this;
        }

        double fps() const {
            return this->m_fps;
        }

        operator double() const { return static_cast<double>(this->fps()); }

        operator float() const { return static_cast<float>(this->fps()); }

        operator int() const { return static_cast<int>(lround(this->fps() + 0.5)); }

        operator long() const { return static_cast<long>(lround(this->fps() + 0.5)); }

        operator unsigned int() const { return static_cast<unsigned int>(lround(this->fps() + 0.5)); }

        operator unsigned long() const { return static_cast<unsigned long>(lround(this->fps() + 0.5)); }

    private:
        seconds m_range = seconds(1);
        std::deque<time_point> m_every_time_point;
        double m_fps = 0;
    };
}



#endif //ORZ_TOOLS_FPS_H
