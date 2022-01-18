//
// Created by Lby on 2017/6/1.
//

#ifndef ORZ_UTILS_FORMAT_H
#define ORZ_UTILS_FORMAT_H

#include <string>
#include <sstream>
#include <vector>
#include <chrono>

#ifndef UNUSED
#define UNUSED(val) (void)(val)
#endif

namespace orz {
    using time_point = decltype(std::chrono::system_clock::now());

    static inline void _Concat_str(std::ostream &out) { (decltype(_Concat_str(out))()); }

    template<typename T, typename... Args>
    static inline void _Concat_str(std::ostream &out, T &&t, Args&&... args) { _Concat_str(out << std::forward<T>(t), std::forward<Args>(args)...); }

    template<typename... Args>
    static inline const std::string Concat(Args&&... args) {
        std::ostringstream oss;
        _Concat_str(oss, std::forward<Args>(args)...);
        return oss.str();
    }

    const std::string Format(const std::string &f);

    std::vector<std::string> Split(const std::string &str, char ch = ' ', size_t _size = 0);

    std::vector<std::string> Split(const std::string &str, const std::string sep = " ", size_t _size = 0);

    std::string Join(const std::vector<std::string> &list, const std::string &sep);

    template<typename T>
    static inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
        out << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i) out << ", ";
            out << vec[i];
        }
        out << "]";
        return out;
    }

    std::string to_string(time_point tp, const std::string &format = "%Y-%m-%d %H:%M:%S");

    /**
     * return format now time string
     * @param format same as strftime
     * @return string contains now time
     * @see strftime
     */
    std::string now_time(const std::string &format = "%Y-%m-%d %H:%M:%S");
}

#endif //ORZ_UTILS_FORMAT_H
