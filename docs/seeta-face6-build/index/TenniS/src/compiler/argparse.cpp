//
// Created by kier on 2019-06-10.
//

#include <compiler/argparse.h>

#include <utils/log.h>

namespace ts {
    void ts::ArgParser::add(const std::vector<std::string> &arg, const std::vector<std::string> &neg_arg,
                            bool default_value) {
        if (arg.empty()) {
            TS_LOG_ERROR << "param@1 can not be empty." << eject;
        }

        const std::string &true_arg_name = arg.front();
        m_true_arg_names.insert(std::make_pair(true_arg_name, true_arg_name));
        for (auto it = arg.begin() + 1; it != arg.end(); ++it) {
            m_true_arg_names.insert(std::make_pair(*it, true_arg_name));
        }
        for (auto it = neg_arg.begin(); it != neg_arg.end(); ++it) {
            m_false_arg_names.insert(std::make_pair(*it, true_arg_name));
        }
        if (default_value) {
            m_arg_value.insert(std::make_pair(true_arg_name, default_value));
        }
    }

    bool ArgParser::set(const std::string &arg) {
        auto true_it = m_true_arg_names.find(arg);
        if (true_it != m_true_arg_names.end()) {
            m_arg_value[true_it->second] = true;
            return true;
        }
        auto false_it = m_false_arg_names.find(arg);
        if (false_it != m_false_arg_names.end()) {
            // m_arg_value.erase(false_it->second);
            m_arg_value[false_it->second] = false;
            return true;
        }
        return false;
    }

    bool ArgParser::get(const std::string &arg) const {
        auto it = m_arg_value.find(arg);
        if (it != m_arg_value.end()) {
            return it->second;
        }
        return false;
    }

    void ArgParser::parse(const std::string &args) {
        auto params = Split(args, " \t\r\n");
        for (auto &param : params) {
            if (param.empty()) continue;
            set(param);
        }
    }
}
