//
// Created by Lby on 2017/6/1.
//

#include "orz/utils/format.h"
#include "orz/utils/platform.h"
#include <ctime>
#include <iomanip>

namespace orz {
    const std::string Format(const std::string &f) {
        return f;
    }

    std::vector<std::string> Split(const std::string &str, char ch, size_t _size) {
        std::vector<std::string> result;
        std::string::size_type left = 0, right;

        result.reserve(_size);
        while (true) {
            right = str.find(ch, left);
            result.push_back(str.substr(left, right == std::string::npos ? std::string::npos : right - left));
            if (right == std::string::npos) break;
            left = right + 1;
        }
        return std::move(result);
    }

    std::string::size_type FindDecollator(const std::string &str, const std::string &sep, std::string::size_type off) {
        if (off == std::string::npos) return std::string::npos;
        std::string::size_type i = off;
        for (; i < str.length(); ++i) {
            if (sep.find(str[i]) != std::string::npos) return i;
        }
        return std::string::npos;
    }

    std::vector<std::string> Split(const std::string &str, const std::string sep, size_t _size) {
        std::vector<std::string> result;
        std::string::size_type left = 0, right;

        result.reserve(_size);
        while (true) {
            right = FindDecollator(str, sep, left);
            result.push_back(str.substr(left, right == std::string::npos ? std::string::npos : right - left));
            if (right == std::string::npos) break;
            left = right + 1;
        }
        return std::move(result);
    }

    std::string Join(const std::vector<std::string>& list, const std::string &sep) {
        std::ostringstream oss;
        for (size_t i = 0; i < list.size(); ++i) {
            if (i) oss << sep;
            oss << list[i];
        }
        return oss.str();
    }

    static struct tm time2tm(std::time_t from) {
        std::tm to = {0};
#if ORZ_PLATFORM_CC_MSVC
        localtime_s(&to, &from);
#else
        localtime_r(&from, &to);
#endif
        return to;
    }

    std::string to_string(time_point tp, const std::string &format) {
        std::time_t tt = std::chrono::system_clock::to_time_t(tp);
        std::tm even = time2tm(tt);
        char tmp[64];
        std::strftime(tmp, sizeof(tmp), format.c_str(), &even);
        return std::string(tmp);
    }

    std::string now_time(const std::string &format) {
        return to_string(std::chrono::system_clock::now(), format);
    }
}
