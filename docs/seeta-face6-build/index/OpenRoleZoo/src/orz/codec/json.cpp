//
// Created by lby on 2018/1/10.
//

#include "orz/codec/json.h"

#include "json_iterator.h"
#include <cctype>
#include <cstdlib>
#include <cstdio>

#include <unordered_map>
#include <orz/mem/need.h>
#include "orz/io/i.h"
#include "orz/codec/base64.h"
#include "orz/io/dir.h"

namespace orz {
    static bool is_space(char ch) {
        return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
    }

    static json_iterator jump_space(json_iterator it) {
        while (it != it.end() && is_space(*it)) ++it;
        return it;
    }

    static bool parse_null(json_iterator &beg, jug &value) {
        beg = jump_space(beg);
        if (beg == beg.end()) ORZ_LOG(ERROR) << "syntax error: converting empty json to null" << crash;
        if (beg.cut(beg + 4) == "null") {
            beg += 4;
            value = nullptr;
            return true;
        }
        return false;
    }

    static jug parse_boolean(json_iterator &beg) {
        beg = jump_space(beg);
        if (beg == beg.end()) ORZ_LOG(ERROR) << "syntax error: converting empty json to boolean" << crash;
        jug result;
        if (beg.cut(beg + 4) == "true") {
            beg += 4;
            result = true;
        } else if (beg.cut(beg + 5) == "false") {
            beg += 5;
            result = false;
        }
        return result;
    }

    int char2hex(char ch) {
        int lch = std::tolower(ch);
        if ('0' <= lch && lch <= '9') return lch - '0';
        if ('a' <= lch && lch <= 'f') return lch - 'a' + 10;
        return -1;
    }

    static std::string parse_string(json_iterator &beg) {
        beg = jump_space(beg);
        if (beg == beg.end()) ORZ_LOG(ERROR) << "syntax error: converting empty json to string" << crash;
        if (*beg != '"') ORZ_LOG(ERROR) << "syntax error: string begin with " << *beg << crash;
        std::string value;
        auto it = beg;
        bool slant = false;
        int unicode_index = 0;
        char unicode = 0;
        while (++it != it.end()) {
            if (unicode_index > 0) {
                int ch = char2hex(*it);
                if (ch < 0) ORZ_LOG(ERROR) << "syntax error: unrecognized unicode" << crash;
                switch (unicode_index) {
                    case 1:
                        unicode |= (ch << 4);
                        unicode_index++;
                        break;
                    case 2:
                        unicode |= ch;
                        value.push_back(char(unicode));
                        unicode = 0;
                        unicode_index++;
                        break;
                    case 3:
                        unicode |= (ch << 4);
                        unicode_index++;
                        break;
                    case 4:
                        unicode |= ch;
                        value.push_back(char(unicode));
                        unicode = 0;
                        unicode_index = 0;
                        break;
                    default:
                        break;
                }
                continue;
            } else if (slant) {
                switch (*it) {
                    case '\"':
                        value.push_back(*it);
                        break;
                    case '\\':
                        value.push_back(*it);
                        break;
                    case '/':
                        value.push_back(*it);
                        break;
                    case 'b':
                        value.push_back('\b');
                        break;
                    case 'f':
                        value.push_back('\f');
                        break;
                    case 'n':
                        value.push_back('\n');
                        break;
                    case 'r':
                        value.push_back('\r');
                        break;
                    case 't':
                        value.push_back('\t');
                        break;
                    case 'u':
                        unicode_index = 1;
                        break;
                    default:
                        value.push_back(*it);
                        break;
                }
                slant = false;
                continue;
            } else if (*it == '\\') {
                slant = true;
                continue;
            } else if (*it == '"') {
                beg = it + 1;
                return std::move(value);
            }
            value.push_back(*it);
        }
        ORZ_LOG(ERROR) << "syntax error: can not find match \"" << crash;
        return std::string();
    }

    static jug parse_number(json_iterator &beg) {
        beg = jump_space(beg);
        if (beg == beg.end()) ORZ_LOG(ERROR) << "syntax error: converting empty json to number" << crash;
        jug result;
        const char *number_c_string = &(*beg);
        char *end_ptr = nullptr;
        double value = std::strtod(number_c_string, &end_ptr);
        if (end_ptr == number_c_string) return result;
        auto ivalue = static_cast<int>(value);
        if (double(ivalue) == value) result = ivalue;
        else result = value;
        beg += int(end_ptr - number_c_string);
        return result;
    }

    static jug parse_value(json_iterator &beg);

    static jug parse_list(json_iterator &beg) {
        beg = jump_space(beg);
        if (beg == beg.end()) ORZ_LOG(ERROR) << "syntax error: converting empty json to list" << crash;
        if (*beg != '[') ORZ_LOG(ERROR) << "syntax error: list begin with " << *beg << crash;
        jug value(Piece::LIST);
        auto it = beg;
        while (++it != it.end()) {
            it = jump_space(it);
            if (it == it.end() || *it == ']') break;
            jug local_value = parse_value(it);
            value.append(local_value);
            it = jump_space(it);
            if (it != it.end() && *it == ',') continue;
            break;
        }
        if (it == it.end() || *it != ']') ORZ_LOG(ERROR) << "syntax error: can not find match ]" << crash;
        beg = it + 1;
        return std::move(value);
    }

    static jug parse_dict(json_iterator &beg) {
        beg = jump_space(beg);
        if (beg == beg.end()) ORZ_LOG(ERROR) << "syntax error: converting empty json to dict" << crash;
        if (*beg != '{') ORZ_LOG(ERROR) << "syntax error: dict begin with " << *beg << crash;
        jug value(Piece::DICT);
        auto it = beg;
        while (++it != it.end()) {
            it = jump_space(it);
            if (it == it.end() || *it == '}') break;
            std::string local_key = parse_string(it);
            it = jump_space(it);
            if (it == it.end() || *it != ':')
                ORZ_LOG(ERROR) << "syntax error: dict key:value must split with :" << crash;
            ++it;
            jug local_value = parse_value(it);
            value.index(local_key, local_value);
            it = jump_space(it);
            if (it != it.end() && *it == ',') continue;
            break;
        }
        if (it == it.end() || *it != '}') ORZ_LOG(ERROR) << "syntax error: can not find match }" << crash;
        beg = it + 1;
        return std::move(value);
    }

    static jug pack_date(const std::vector<std::string> &args) {
        return now_time("%Y-%m-%d");
    }

    static jug pack_time(const std::vector<std::string> &args) {
        return now_time("%H:%M:%S");
    }

    static jug pack_datetime(const std::vector<std::string> &args) {
        return now_time("%Y-%m-%d %H:%M:%S");
    }

    static jug pack_nil(const std::vector<std::string> &args) {
        return jug(nullptr);
    }

    static jug pack_error(const std::vector<std::string> &args) {
        ORZ_LOG(ERROR) << "Not supported command: " << args[0] << crash;
        return jug(nullptr);
    }

    static jug pack_file(const std::vector<std::string> &args) {
        if (args.size() < 2) {
            ORZ_LOG(ERROR) << "Command format error, should be @file@..." << crash;
        }
        auto data = read_file(args[1]);
        if (data.empty()) {
            ORZ_LOG(ERROR) << args[1] << " is not a valid file." << crash;
        }
        return data;
    }

    static jug pack_base64(const std::vector<std::string> &args) {
        if (args.size() < 2) {
            ORZ_LOG(ERROR) << "Command format error, should be @base64@..." << crash;
        }
        auto data = base64_decode(args[1]);
        return binary(data.data(), data.size());
    }

    using command_handler = std::function<jug(const std::vector<std::string> &args)>;

    static command_handler registered_command(const std::string &command) {
        static std::unordered_map<std::string, command_handler> command_map = {
                {"date", pack_date},
                {"time", pack_time},
                {"datetime", pack_datetime},
                {"nil", pack_nil},
                {"binary", pack_error},
                {"file", pack_file},
                {"base64", pack_base64},
        };
        auto it = command_map.find(command);
        if (it == command_map.end()) return nullptr;
        return it->second;
    }

    static jug parse_sta_string(json_iterator &beg) {
        auto str = parse_string(beg);
        if (str.empty() || str[0] != '@') return str;
        auto key_args = Split(str, '@');
        auto &key = key_args[1];
        auto args = std::vector<std::string>(key_args.begin() + 1, key_args.end());
        auto commond = registered_command(key);
        if (commond == nullptr) return str;
        return commond(args);
    }

    /// TODO: add bool support
    static jug parse_value(json_iterator &beg) {
        beg = jump_space(beg);
        if (beg == beg.end()) ORZ_LOG(ERROR) << "syntax error: converting empty json" << crash;
        jug value;
        auto it = beg;
        value = parse_number(beg);
        if (value.valid()) return value;
        if (*it == '"') return parse_sta_string(beg);
        if (*it == '[') return parse_list(beg);
        if (*it == '{') return parse_dict(beg);
        value = parse_boolean(beg);
        if (value.valid()) return value;
        if (parse_null(beg, value)) return value;
        ORZ_LOG(ERROR) << "syntax error: unrecognized symbol " << *it << crash;
        return jug();
    }

    jug json2jug(const std::string &json) {
        try {
            json_iterator it(json.c_str(), static_cast<int>(json.length()));
            return parse_value(it);
        } catch (const Exception &) {
            return orz::jug();
        }

    }

    std::string jug2json(const orz::jug &obj) {
        return obj.repr();
    }

    static bool is_alphanumeric(char ch) {
        if ('a' <= ch && ch <= 'z') return true;
        if ('A' <= ch && ch <= 'Z') return true;
        if ('0' <= ch && ch <= '9') return true;
        return false;
    }

    static std::string HH(unsigned char ch) {
        char buff[3];
#if _MSC_VER >= 1600
		sprintf_s(buff, "%02X", ch);
#else
        std::sprintf(buff, "%02X", ch);
#endif
        return std::string(buff, buff + 2);
    }

    static std::string form_encode(const std::string &str) {
        std::ostringstream oss;
        for (auto ch : str) {
            if (ch == ' ') oss << char('+');
            else if (is_alphanumeric(ch)) oss << char(ch);
            else oss << '%' << HH(ch);
        }
        return oss.str();
    }

    std::string form_encode(const orz::jug &obj) {
        if (!obj.valid(Piece::DICT)) ORZ_LOG(ERROR) << "form encoding only supporting dict" << crash;
        std::ostringstream oss;
        int first = true;
        for (auto &key : obj.keys()) {
            if (first) first = false;
            else oss << "&";
            oss << key << "=" << form_encode(obj[key].str());
        }
        return oss.str();
    }

    jug json2jug(const std::string &json, const std::string &root) {
        auto dir = cut_path_tail(root);
        auto cur = getcwd();
        need pop_dir([&](){ cd(cur);});
        cd(dir);
        return json2jug(json);
    }

}
