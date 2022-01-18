//
// Created by kier on 2019-06-10.
//

#ifndef TENSORSTACK_COMPILER_ARGPARSE_H
#define TENSORSTACK_COMPILER_ARGPARSE_H

#include <string>
#include <vector>
#include <map>

namespace ts {
    class ArgParser {
    public:
        void add(const std::vector<std::string> &arg, const std::vector<std::string> &neg_arg, bool default_value = false);
        void parse(const std::string &args);

        bool get(const std::string &arg) const;

        /**
         *
         * @param arg
         * @return false if not arg added
         */
        bool set(const std::string &arg);

    private:
        std::map<std::string, std::string> m_true_arg_names;
        std::map<std::string, std::string> m_false_arg_names;
        std::map<std::string, bool> m_arg_value;
    };
}


#endif //TENSORSTACK_COMPILER_ARGPARSE_H
