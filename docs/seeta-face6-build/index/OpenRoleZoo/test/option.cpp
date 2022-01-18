//
// Created by kier on 2018/9/25.
//

#include <orz/tools/option.h>
#include <orz/utils/log.h>

template <typename T>
static std::string to_string(const std::set<T> &_set,
        const std::string &prefix = "",
        const std::string &suffix = "") {
    bool first = true;
    std::ostringstream oss;
    oss << "[";
    for (auto &_item : _set) {
        if (first) {
            first = false;
        } else {
            oss << ", ";
        }
        oss << prefix << _item << suffix;
    }
    oss << "]";
    return oss.str();
}

int main() {
    orz::arg::OptionSet option;
    auto pose = option.add(orz::arg::INT, "p")->
            property(orz::arg::REQUIRED)->
            value(25)->
            description("Set pose, example: 1, 2...");
    auto head = option.add(orz::arg::INT, "h")->
            property(orz::arg::REQUIRED)->
            value(25)->
            description("Set head, example: 1, 2...");

    if (!option.parse("-p=64")) {
        ORZ_LOG(orz::INFO) << option.last_error_message();
    }

    for (auto opt : option) {
        ORZ_LOG(orz::INFO) << to_string(opt.name(), "-") << (opt.found() ? "(set)" : "(not set)") << ": "
                           << opt.value().to_string();
    }
}