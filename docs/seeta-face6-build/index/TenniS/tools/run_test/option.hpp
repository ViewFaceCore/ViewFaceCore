//
// Created by kier on 2019/3/4.
//

#ifndef TENSORSTACK_OPTION_HPP
#define TENSORSTACK_OPTION_HPP

#include <string>
#include <memory>
#include <iostream>
#include <iomanip>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#include <cstdint>
#include <climits>
#include <cctype>
#include <algorithm>

namespace ts {
    namespace arg {

        enum ValueType {
            STRING = 1,
            FLOAT = 2,
            INT = 3,
            BOOLEAN = 4,
        };

        template<ValueType _TYPE>
        struct Type;

        template<>
        struct Type<STRING> {
            using declare = std::string;
            static const declare default_value;
        };

        template<>
        struct Type<FLOAT> {
            using declare = float;
            static const declare default_value;
        };
        template<>
        struct Type<INT> {
            using declare = long;
            static const declare default_value;
        };
        template<>
        struct Type<BOOLEAN> {
            using declare = bool;
            static const declare default_value;
        };

        const Type<STRING>::declare Type<STRING>::default_value = std::string("");
        const Type<FLOAT>::declare Type<FLOAT>::default_value = 0.0f;
        const Type<INT>::declare Type<INT>::default_value = 0;
        const Type<BOOLEAN>::declare Type<BOOLEAN>::default_value = false;

        class Value {
        public:
            using self = Value;

            explicit Value(ValueType type) : m_type(type) {}

            virtual ~Value() = default;

            ValueType type() { return m_type; }

        private:
            ValueType m_type;
        };

        template<ValueType _TYPE>
        class ValueWithType : public Value {
        public:
            using self = ValueWithType;
            using supper = Value;

            explicit ValueWithType() : supper(_TYPE) {}
        };

        template<ValueType _TYPE>
        class ValueDefinition : public ValueWithType<_TYPE> {
        public:
            using self = ValueDefinition;
            using supper = ValueWithType<_TYPE>;

            using declare = typename Type<_TYPE>::declare;

            ValueDefinition() = default;

            ValueDefinition(const declare &value) : m_value(value) {}

            ~ValueDefinition() = default;

            void set(const declare &value) { m_value = value; }

            declare get() const { return m_value; }

            operator declare() const { return m_value; }

        private:
            declare m_value = Type<_TYPE>::default_value;
        };

        using ValueString   = ValueDefinition<STRING>;
        using ValueFloat    = ValueDefinition<FLOAT>;
        using ValueInt      = ValueDefinition<INT>;
        using ValueBoolean  = ValueDefinition<BOOLEAN>;

        static inline std::shared_ptr<Value> make_shared(ValueType type) {
            switch (type) {
                default:
                    return nullptr;
                case STRING:
                    return std::make_shared<ValueDefinition<STRING>>();
                case FLOAT:
                    return std::make_shared<ValueDefinition<FLOAT>>();
                case INT:
                    return std::make_shared<ValueDefinition<INT>>();
                case BOOLEAN:
                    return std::make_shared<ValueDefinition<BOOLEAN>>();
            }
        }

        static inline std::string tolower(const std::string &str) {
            std::string str_copy = str;
            for (auto &ch : str_copy) {
                ch = char(std::tolower(ch));
            }
            return str_copy;
        }

        class ValueCommon {
        public:
            using self = ValueCommon;

            ValueCommon() {}

            ValueCommon(const Type<STRING>::declare &value)
                    : m_value(new ValueDefinition<STRING>(value)) {}

            ValueCommon(Type<FLOAT>::declare value)
                    : m_value(new ValueDefinition<FLOAT>(value)) {}

            ValueCommon(Type<INT>::declare value)
                    : m_value(new ValueDefinition<INT>(value)) {}

            ValueCommon(Type<BOOLEAN>::declare value)
                    : m_value(new ValueDefinition<BOOLEAN>(value)) {}

            ValueCommon(ValueType type)
                    : m_value(make_shared(type)) {}

            bool valid() const { return m_value != nullptr; }

            bool valid(ValueType type) { return m_value != nullptr && m_value->type() == type; }

            int type() const {
                if (m_value == nullptr) return 0;
                return int(m_value->type());
            }

            self &reset() {
                m_value.reset();
                return *this;
            }

            self &reset(ValueType type) {
                m_value.reset();
                switch (type) {
                    default:
                        break;
                    case STRING:
                        m_value.reset(new ValueDefinition<STRING>);
                        break;
                    case FLOAT:
                        m_value.reset(new ValueDefinition<FLOAT>);
                        break;
                    case INT:
                        m_value.reset(new ValueDefinition<INT>);
                        break;
                    case BOOLEAN:
                        m_value.reset(new ValueDefinition<BOOLEAN>);
                        break;
                }
                return *this;
            }

            bool set(const Type<STRING>::declare &value) {
                if (m_value == nullptr) {
                    m_value.reset(new ValueDefinition<STRING>(value));
                    return true;
                }
                switch (m_value->type()) {
                    default:
                        return false;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        self_value->set(value);
                        return true;
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        self_value->set(Type<FLOAT>::declare(std::atof(value.c_str())));
                        return true;
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        self_value->set(Type<INT>::declare(std::atol(value.c_str())));
                        return true;
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        auto lower_value = tolower(value);
                        self_value->set(lower_value == "true" || lower_value == "on");
                        return true;
                    }
                }
                return false;
            }

            bool set(Type<FLOAT>::declare value) {
                if (m_value == nullptr) {
                    m_value.reset(new ValueDefinition<FLOAT>(value));
                    return true;
                }
                switch (m_value->type()) {
                    default:
                        return false;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        self_value->set(std::to_string(value));
                        return true;
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        self_value->set(Type<FLOAT>::declare(value));
                        return true;
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        self_value->set(Type<INT>::declare(value));
                        return true;
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        self_value->set(value != 0.0f);
                        return true;
                    }
                }
                return false;
            }

            bool set(Type<INT>::declare value) {
                if (m_value == nullptr) {
                    m_value.reset(new ValueDefinition<INT>(value));
                    return true;
                }
                switch (m_value->type()) {
                    default:
                        return false;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        self_value->set(std::to_string(value));
                        return true;
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        self_value->set(Type<FLOAT>::declare(value));
                        return true;
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        self_value->set(Type<INT>::declare(value));
                        return true;
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        self_value->set(value != 0);
                        return true;
                    }
                }
                return false;
            }

            bool set(Type<BOOLEAN>::declare value) {
                if (m_value == nullptr) {
                    m_value.reset(new ValueDefinition<BOOLEAN>(value));
                    return true;
                }
                switch (m_value->type()) {
                    default:
                        return false;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        self_value->set(value ? "true" : "false");
                        return true;
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        self_value->set(Type<FLOAT>::declare(value));
                        return true;
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        self_value->set(Type<INT>::declare(value));
                        return true;
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        self_value->set(value);
                        return true;
                    }
                }
                return false;
            }

            bool set(const char *value) { return set(Type<STRING>::declare(value)); }

            bool set(double value) { return set(Type<FLOAT>::declare(value)); }

            bool set(char value) { return set(Type<INT>::declare(value)); }

            bool set(unsigned char value) { return set(Type<INT>::declare(value)); }

            bool set(short value) { return set(Type<INT>::declare(value)); }

            bool set(unsigned short value) { return set(Type<INT>::declare(value)); }

            bool set(int value) { return set(Type<INT>::declare(value)); }

            bool set(unsigned int value) { return set(Type<INT>::declare(value)); }

            bool set(unsigned long value) { return set(Type<INT>::declare(value)); }

            Type<STRING>::declare to_string() const {
                if (m_value == nullptr) {
                    return Type<STRING>::default_value;
                }
                switch (m_value->type()) {
                    default:
                        return Type<STRING>::default_value;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        return self_value->get();
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        return std::to_string(self_value->get());
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        return std::to_string(self_value->get());
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        return self_value->get() ? "true" : "false";
                    }
                }
            }

            Type<FLOAT>::declare to_float() const {
                if (m_value == nullptr) {
                    return Type<FLOAT>::default_value;
                }
                switch (m_value->type()) {
                    default:
                        return Type<FLOAT>::default_value;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        return Type<FLOAT>::declare(std::atof(self_value->get().c_str()));
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        return Type<FLOAT>::declare(self_value->get());
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        return Type<FLOAT>::declare(self_value->get());
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        return Type<FLOAT>::declare(self_value->get());
                    }
                }
            }

            Type<INT>::declare to_int() const {
                if (m_value == nullptr) {
                    return Type<INT>::default_value;
                }
                switch (m_value->type()) {
                    default:
                        return Type<INT>::default_value;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        return Type<INT>::declare(std::atol(self_value->get().c_str()));
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        return Type<INT>::declare(self_value->get());
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        return Type<INT>::declare(self_value->get());
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        return Type<INT>::declare(self_value->get());
                    }
                }
            }

            Type<BOOLEAN>::declare to_boolean() const {
                if (m_value == nullptr) {
                    return Type<BOOLEAN>::default_value;
                }
                switch (m_value->type()) {
                    default:
                        return Type<BOOLEAN>::default_value;
                    case STRING: {
                        auto self_value = dynamic_cast<ValueDefinition<STRING> *>(m_value.get());
                        auto value = self_value->get();
                        auto lower_value = tolower(value);
                        return lower_value == "true" || lower_value == "on";
                    }
                    case FLOAT: {
                        auto self_value = dynamic_cast<ValueDefinition<FLOAT> *>(m_value.get());
                        return self_value->get() != 0.0f;
                    }
                    case INT: {
                        auto self_value = dynamic_cast<ValueDefinition<INT> *>(m_value.get());
                        return self_value->get() != 0;
                    }
                    case BOOLEAN: {
                        auto self_value = dynamic_cast<ValueDefinition<BOOLEAN> *>(m_value.get());
                        return self_value->get();
                    }
                }
            }

            operator std::string() const { return to_string(); }

            operator float() const { return (float) (to_float()); }

            operator int() const { return (int) (to_int()); }

            operator bool() const { return (bool) (to_boolean()); }

            operator double() const { return (double) (to_float()); }

            operator char() const { return (char) (to_int()); }

            operator unsigned char() const { return (unsigned char) (to_int()); }

            operator short() const { return (short) (to_int()); }

            operator unsigned short() const { return (unsigned short) (to_int()); }

            operator unsigned int() const { return (unsigned int) (to_int()); }

            operator long() const { return (long) (to_int()); }

            operator unsigned long() const { return (unsigned long) (to_int()); }

            template<ValueType _TYPE>
            operator ValueDefinition<_TYPE>() {
                if (this->valid(_TYPE)) return *dynamic_cast<ValueDefinition<_TYPE> *>(this->m_value.get());
                return ValueDefinition<_TYPE>();
            }

            friend std::ostream &operator<<(std::ostream &out, const self &object);

        private:
            std::shared_ptr<Value> m_value;
        };

        inline std::ostream &operator<<(std::ostream &out, const ValueCommon &object) {
            switch (object.type()) {
                default:
                    return out;
                case STRING:
                    return out << object.to_string();
                case FLOAT:
                    return out << object.to_float();
                case INT:
                    return out << object.to_int();
                case BOOLEAN:
                    return out << (object.to_boolean() ? "true" : "false");
            }
            return out;
        }

        enum OptionProperty {
            REQUIRED = 0x1,
            OPTIONAL = 0x1 << 1,
        };

        class Option {
        public:
            using self = Option;

            static const char prefix;

            self *property(OptionProperty _prop) {
                m_prop = _prop;
                return this;
            }

            OptionProperty property() const { return m_prop; }

            self *name(const std::string &_name) {
                return this->name(std::set<std::string>(&_name, &_name + 1));
            }

            self *name(const std::set<std::string> &_name) {
                m_names = _name;
                return this;
            }

            self *name(std::set<std::string> &&_name) {
                m_names = std::forward<std::set<std::string>>(_name);
                return this;
            }

            self *name(const std::initializer_list<std::string> &_string_list) {
                return this->name(std::set<std::string>(_string_list.begin(), _string_list.end()));
            }

            const std::set<std::string> &name() const { return m_names; }

            self *description(const std::string &_desc) {
                m_description = _desc;
                return this;
            }

            const std::string &description() const { return m_description; }

            self *type(ValueType _type) {
                m_value.reset(_type);
                return this;
            }

            int type() { return m_value.type(); }

            self *found(bool _found) {
                m_found = _found;
                return this;
            }

            bool found() const { return m_found; }

            self *value(const std::string &_value) {
                m_value.set(_value);
                return this;
            }

            self *value(const char *_value) {
                m_value.set(_value);
                return this;
            }

            self *value(float _value) {
                m_value.set(_value);
                return this;
            }

            self *value(double _value) {
                m_value.set(_value);
                return this;
            }

            self *value(char _value) {
                m_value.set(_value);
                return this;
            }

            self *value(unsigned char _value) {
                m_value.set(_value);
                return this;
            }

            self *value(short _value) {
                m_value.set(_value);
                return this;
            }

            self *value(unsigned short _value) {
                m_value.set(_value);
                return this;
            }

            self *value(int _value) {
                m_value.set(_value);
                return this;
            }

            self *value(unsigned int _value) {
                m_value.set(_value);
                return this;
            }

            self *value(long _value) {
                m_value.set(_value);
                return this;
            }

            self *value(unsigned long _value) {
                m_value.set(_value);
                return this;
            }

            const ValueCommon &value() { return m_value; }

            bool match(const std::string &_name) const {
                return m_names.find(_name) != m_names.end();
            }

            bool parse(const std::string &arg) {
                // step 0: get name and value
                std::string name, value;
                split(arg, name, value);
                return parse(name, value);
            }

            bool parse(const std::string &name, const std::string &value) {
                if (name.empty()) return false;
                // step 1: check if name matched
                if (!match(name)) return false;

                if (this->type() != BOOLEAN && value.empty()) {
                    return false;
                }

                // step 2: set value
                if (this->type() == BOOLEAN && value.empty()) {
                    m_value.set(true);
                } else {
                    m_value.set(value);
                }

                m_found = true;
                return true;
            }

            static void split(const std::string &arg, std::string &name, std::string &value) {
                if (arg.empty() || arg[0] != prefix) return;
                name.clear();
                value.clear();
                auto equal_sign_i = arg.find('=');
                if (equal_sign_i == std::string::npos) {
                    name = arg.substr(1);
                } else {
                    name = arg.substr(1, equal_sign_i - 1);
                    value = arg.substr(equal_sign_i + 1);
                }
            }

            friend std::ostream &operator<<(std::ostream &out, const self &object);

        private:
            OptionProperty m_prop = OPTIONAL;
            std::set<std::string> m_names;
            std::string m_description;
            ValueCommon m_value;
            bool m_found = false;
        };

        inline std::ostream &operator<<(std::ostream &out, const Option &option) {
            bool first_name = true;
            out << "[";
            for (auto &name : option.name()) {
                if (first_name) {
                    first_name = false;
                } else {
                    out << " ";
                }
                out << "-" << name;
            }
            out << "]";
            if (option.property() == REQUIRED) {
                out << " [REQUIRED]";
            }
            if (!option.description().empty()) {
                out << ": " << option.description();
            }
            return out;
        }

        const char Option::prefix = '-';

        template<typename T>
        T min(T a, T b, T c) {
            return std::min<T>(std::min<T>(a, b), c);
        }

        static inline int edit_distance(const std::string &lhs, const std::string &rhs) {
            const size_t M = lhs.length();  // rows
            const size_t N = rhs.length();  // cols

            if (M == 0) return int(N);
            if (N == 0) return int(M);

            std::unique_ptr<int[]> dist(new int[M * N]);
#define __EDIT_DIST(m, n) (dist[(m) * N + (n)])
            __EDIT_DIST(0, 0) = lhs[0] == rhs[0] ? 0 : 2;
            for (size_t n = 1; n < N; ++n) {
                __EDIT_DIST(0, n) = __EDIT_DIST(0, n - 1) + 1;
            }
            for (size_t m = 1; m < M; ++m) {
                __EDIT_DIST(m, 0) = __EDIT_DIST(m - 1, 0) + 1;
            }
            for (size_t m = 1; m < M; ++m) {
                for (size_t n = 1; n < N; ++n) {
                    if (lhs[m] == rhs[n]) {
                        __EDIT_DIST(m, n) = min(
                                __EDIT_DIST(m - 1, n),
                                __EDIT_DIST(m, n - 1),
                                __EDIT_DIST(m - 1, n - 1));
                    } else {
                        __EDIT_DIST(m, n) = min(
                                __EDIT_DIST(m - 1, n) + 1,
                                __EDIT_DIST(m, n - 1) + 1,
                                __EDIT_DIST(m - 1, n - 1) + 2);
                    }
                }
            }
            return dist[M * N - 1];
#undef __EDIT_DIST
        }

        class OptionSet {
        public:
            using self = OptionSet;

            Option *add(ValueType type, const std::initializer_list<std::string> &list) {
                std::shared_ptr<Option> option(new Option);
                option->type(type)->name(list);
                return this->add(option);
            }

            Option *add(ValueType type, const std::string &name) {
                std::shared_ptr<Option> option(new Option);
                option->type(type)->name(name);
                return this->add(option);
            }

            Option *add(ValueType type, const std::set<std::string> &name) {
                std::shared_ptr<Option> option(new Option);
                option->type(type)->name(name);
                return this->add(option);
            }

            Option *add(ValueType type, std::set<std::string> &&name) {
                std::shared_ptr<Option> option(new Option);
                option->type(type)->name(name);
                return this->add(option);
            }

            Option *add(const std::shared_ptr<Option> &option) {
                std::shared_ptr<Option> option_copy = option;
                m_option_list.push_back(option);
                for (auto name : option_copy->name()) {
                    m_options.insert(std::make_pair(name, option_copy));
                }
                return option_copy.get();
            }

            void clear() {
                m_options.clear();
            }

            std::string last_error_message() const {
                return m_last_error_message;
            }

            bool parse(const std::string &arg) {
                // list type of errors, check last_error_message
                // unrecognized option
                // option format error
                if (arg.empty() || arg[0] != Option::prefix) {
                    std::ostringstream oss;
                    oss << "Argument option must start with prefix " << Option::prefix;
                    m_last_error_message = oss.str();
                    return false;
                }

                std::string name, value;
                Option::split(arg, name, value);
                auto option_it = m_options.find(name);
                if (option_it == m_options.end()) {
                    std::ostringstream oss;
                    std::string fuzzy_name = this->fuzzy_name(name);
                    oss << "Unrecognized option: -" << name;
                    if (!fuzzy_name.empty()) {
                        oss << ", did you mean -" << fuzzy_name << " ?";
                    }
                    m_last_error_message = oss.str();
                    return false;
                }
                if (!option_it->second->parse(name, value)) {
                    std::ostringstream oss;
                    oss << "UnInterpretable option: " << arg;
                    m_last_error_message = oss.str();
                    return false;
                }
                return true;
            }

            bool parse(std::vector<std::string> &args) {
                auto args_copy = args;
                auto arg_it = args_copy.begin();
                while (arg_it != args_copy.end()) {
                    auto arg = *arg_it;
                    if (arg.empty() || arg[0] != Option::prefix) {
                        ++arg_it;
                        continue;
                    }
                    if (parse(arg)) {
                        arg_it = args_copy.erase(arg_it);
                    } else {
                        return false;
                    }
                }
                args = std::move(args_copy);
                return true;
            }

            std::string fuzzy_name(const std::string &name) const {
                if (m_options.empty()) return "";
                int min_edit_distance = INT_MAX;
                std::string closest_name;
                for (auto &name_option_pair : m_options) {
                    auto &target_name = name_option_pair.first;
                    int dist = edit_distance(name, target_name);
                    if (dist < min_edit_distance) {
                        closest_name = target_name;
                        min_edit_distance = dist;
                    }
                }
                return closest_name;
            }

            bool check() const {
                for (auto &option : m_options) {
                    auto &opt = *option.second;
                    if (opt.property() == REQUIRED && !opt.found()) {
                        std::ostringstream oss;
                        oss << "Must set option: " << opt;
                        m_last_error_message = oss.str();
                        return false;
                    }
                }
                return true;
            }

            class option_iterator {
            public:
                using self = option_iterator;

                using inner_iterator = std::vector<std::shared_ptr<Option>>::iterator;

                option_iterator() = default;

                explicit option_iterator(inner_iterator iter) : m_iter(iter) {}

                self &operator++() {
                    this->m_iter++;
                    return *this;
                }

                const self operator++(int) {
                    return self(this->m_iter++);
                }

                self &operator--() {
                    this->m_iter--;
                    return *this;
                }

                const self operator--(int) {
                    return self(this->m_iter--);
                }

                bool operator==(const self &rhs) {
                    return m_iter == rhs.m_iter;
                }

                bool operator!=(const self &rhs) {
                    return m_iter != rhs.m_iter;
                }

                Option &operator*() {
                    return **m_iter;
                }

                Option *operator->() {
                    return (*m_iter).get();
                }

            private:
                inner_iterator m_iter;
            };

            class const_option_iterator {
            public:
                using self = const_option_iterator;

                using inner_iterator = std::vector<std::shared_ptr<Option>>::const_iterator;

                const_option_iterator() = default;

                explicit const_option_iterator(inner_iterator iter) : m_iter(iter) {}

                self &operator++() {
                    this->m_iter++;
                    return *this;
                }

                const self operator++(int) {
                    return self(this->m_iter++);
                }

                self &operator--() {
                    this->m_iter--;
                    return *this;
                }

                const self operator--(int) {
                    return self(this->m_iter--);
                }

                bool operator==(const self &rhs) {
                    return m_iter == rhs.m_iter;
                }

                bool operator!=(const self &rhs) {
                    return m_iter != rhs.m_iter;
                }

                const Option &operator*() {
                    return **m_iter;
                }

                const Option *operator->() {
                    return (*m_iter).get();
                }

            private:
                inner_iterator m_iter;

            };

            option_iterator begin() { return option_iterator(m_option_list.begin()); }

            option_iterator end() { return option_iterator(m_option_list.end()); }

            const_option_iterator begin() const { return const_option_iterator(m_option_list.cbegin()); }

            const_option_iterator end() const { return const_option_iterator(m_option_list.cend()); }

            const_option_iterator cbegin() const { return const_option_iterator(m_option_list.cbegin()); }

            const_option_iterator cend() const { return const_option_iterator(m_option_list.cend()); }

        private:
            std::map<std::string, std::shared_ptr<Option>> m_options;
            std::vector<std::shared_ptr<Option>> m_option_list;
            mutable std::string m_last_error_message;
        };
    }
}


#endif //TENSORSTACK_OPTION_HPP
