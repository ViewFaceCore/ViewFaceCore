//
// Created by kier on 2019/1/29.
//

#include "board/profiler.h"
#include <chrono>
#include <board/profiler.h>
#include <iomanip>
#include <memory>
#include <cstdio>
#include <sstream>

#include "utils/ctxmgr_lite_support.h"

#if _MSC_VER > 1600
#define snprintf sprintf_s
#endif

namespace ts {

    int32_t Profiler::serial_of(const std::string &name) {
        auto it = this->m_serial.find(name);
        if (it != this->m_serial.end()) return it->second;
        auto next_serial = int32_t(this->m_serial.size() + 1);
        this->m_serial.insert(std::make_pair(name, next_serial));
        return next_serial;
    }

    Later Profiler::timer(const std::string &name) {
        using namespace std::chrono;
        auto _start = system_clock::now();
        Later _action([=]() -> void{
            auto _end = system_clock::now();
            auto _duration = std::chrono::duration_cast<microseconds>(_end - _start);
            this->m_board.append(name, _duration.count() / 1000.0f);
        });
        return std::move(_action);
    }

    Board<float> &Profiler::board() {
        return m_board;
    }

    const Board<float> &Profiler::board() const {
        return m_board;
    }

    template <typename T>
    static std::string to_string(const T* ptr) {
        std::ostringstream oss;
        oss << "0x" << std::hex << std::setfill('0') << std::setw(sizeof(T*) * 2) << uint64_t(ptr);
        return oss.str();
    }

    void Profiler::log(std::ostream &out) const {
        std::map<Board<float>::key_type, Board<float>::value_type>
                sorted_board(this->board().begin(), this->board().end());
        out << "============= " << "Profiler(" << to_string(this) << ")" << " timer" << " =============" << std::endl;
        for (auto &key_value : sorted_board) {
            out << "[" << key_value.first << "]: avg spent = " << key_value.second.avg() << "ms" << std::endl;
        }
    }

    Later profiler_timer(const std::string &name) {
        auto profiler = ctx::ptr<Profiler>();
        if (!profiler) return Later();
        return profiler->timer(name);
    }

    bool profiler_on() {
        return ctx::ptr<Profiler>() != nullptr;
    }

    Later profiler_serial_timer(const std::string &name) {
        auto profiler = ctx::ptr<Profiler>();
        if (!profiler) return Later();
        auto size = name.length() * 2 + 1;
        std::unique_ptr<char[]> buffer(new char[size]);
        using namespace std;
        snprintf(buffer.get(), size, name.c_str(), profiler->serial_of(name));
        return profiler->timer(buffer.get());
    }
}

TS_LITE_CONTEXT(ts::Profiler);
