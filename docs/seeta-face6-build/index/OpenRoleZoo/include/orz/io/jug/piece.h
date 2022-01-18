//
// Created by Lby on 2017/8/16.
//

#ifndef ORZ_IO_JUG_PIECE_H
#define ORZ_IO_JUG_PIECE_H

#include "orz/utils/except.h"

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <sstream>

#include "binary.h"

namespace orz {

    class Piece {
    public:
        using self = Piece;

        enum Type {
            NIL = 0,
            INT = 1,
            FLOAT = 2,
            STRING = 3,
            BINARY = 4,
            LIST = 5,
            DICT = 6,
            BOOLEAN = 7
        };

        Piece(Type type)
                : m_type(type) {}

        virtual ~Piece() {}

        virtual std::istream &read(std::istream &bin) = 0;

        virtual std::ostream &write(std::ostream &bin) const = 0;

        static inline void Write(std::ostream &bin, const Piece &pie);

        static inline void Write(std::ostream &bin, const std::shared_ptr<Piece> &pie);

        static inline std::shared_ptr<Piece> Read(std::istream &bin);

        static inline std::shared_ptr<Piece> Get(Type type);

        static inline std::shared_ptr<Piece> Get(Type type, std::istream &bin);

        virtual const std::string str() const {
            std::stringstream oss;
            oss << "Piece<0x" << std::hex << *this << ">";
            return oss.str();
        }

        virtual const std::string repr() const {
            std::stringstream oss;
            oss << "Piece<0x" << std::hex << *this << ">";
            return oss.str();
        }

    public:

        Type type() const {
            return m_type;
        }

        operator bool() const {
            return this->m_type != NIL;
        }

        bool nil() const {
            return this->m_type == NIL;
        }

        bool notnil() const {
            return this->m_type != NIL;
        }

    private:
        Type m_type;
    };

    template<Piece::Type _type>
    class TypedPiece : public Piece {
    public:
        using supper = Piece;
        using self = TypedPiece;

        TypedPiece() : Piece(_type) {}
    };

    template<typename T>
    T defualt_value() {
        return T();
    }

    template<typename T>
    class binio {
    public:
        static std::ostream &write(std::ostream &bin, const T &elem) {
            /// TODO: check big or little endian
            return bin.write(reinterpret_cast<const char *>(&elem), sizeof(T));
        }

        static std::istream &read(std::istream &bin, T &elem) {
            /// TODO: check big or little endian
            return bin.read(reinterpret_cast<char *>(&elem), sizeof(T));
        }
    };

    template<>
    class binio<std::string> {
    public:
        static std::ostream &write(std::ostream &bin, const std::string &str) {
            int size = static_cast<int>(str.size());
            binio<int>::write(bin, size);
            bin.write(str.data(), str.size());
            return bin;
        }

        static std::istream &read(std::istream &bin, std::string &str) {
            int size;
            binio<int>::read(bin, size);
            std::unique_ptr<char[]> buffer(new char[size]);
            bin.read(buffer.get(), size);
            str = std::string(buffer.get(), size);
            return bin;
        }
    };

    template<>
    class binio<binary> {
    public:
        static std::ostream &write(std::ostream &bin, const binary &str) {
            int size = static_cast<int>(str.size());
            binio<int>::write(bin, size);
            bin.write(str.data<char>(), str.size());
            return bin;
        }

        static std::istream &read(std::istream &bin, binary &str) {
            int size;
            binio<int>::read(bin, size);
            str.resize(size);
            bin.read(str.data<char>(), str.size());
            return bin;
        }
    };

    template<Piece::Type _type, typename T>
    class ValuedPiece : public TypedPiece<_type> {
    public:
        using supper = TypedPiece<_type>;
        using self = ValuedPiece;

        using Value = T;

        ValuedPiece() : m_val(defualt_value<T>()) {}

        ValuedPiece(const T &val) : m_val(val) {}

        ValuedPiece(std::istream &bin) {
            this->read(bin);
        }

        ValuedPiece &set(const T &val) {
            this->m_val = val;
            return *this;
        }

        T &get() {
            return this->m_val;
        }

        const T &get() const {
            return this->m_val;
        }

        std::istream &read(std::istream &bin) override {
            return binio<T>::read(bin, m_val);
        }

        std::ostream &write(std::ostream &bin) const override {
            binio<char>::write(bin, static_cast<char>(this->type()));
            binio<T>::write(bin, m_val);
            return bin;
        }

        const std::string str() const override {
            std::stringstream oss;
            oss << m_val;
            return oss.str();
        }

        const std::string repr() const override {
            std::stringstream oss;
            oss << m_val;
            return oss.str();
        }

    private:
        T m_val;
    };

    class NilPiece : public ValuedPiece<Piece::NIL, char> {
    public:
        using supper = ValuedPiece<Piece::NIL, char>;
        using self = NilPiece;

        NilPiece() : supper() {}

        NilPiece(const supper::Value &val) : supper(val) {}

        NilPiece(std::istream &bin) : supper(bin) {}

        const std::string str() const override {
            std::stringstream oss;
            oss << "null";
            return oss.str();
        }

        const std::string repr() const override {
            std::stringstream oss;
            oss << "null";
            return oss.str();
        }
    };

    using IntPiece = ValuedPiece<Piece::INT, int>;

    using FloatPiece = ValuedPiece<Piece::FLOAT, float>;

    class StringPiece : public ValuedPiece<Piece::STRING, std::string> {
    public:
        using supper = ValuedPiece<Piece::STRING, std::string>;
        using self = StringPiece;

        StringPiece() : supper() {}

        StringPiece(const supper::Value &val) : supper(val) {}

        StringPiece(std::istream &bin) : supper(bin) {}

        const std::string repr() const override {
            std::stringstream oss;
            oss << '\"' << get() << '\"';
            return oss.str();
        }
    };

    // using BinaryPiece = ValuedPiece<Piece::BINARY, std::string>;
    class BooleanPiece : public ValuedPiece<Piece::BOOLEAN, char> {
    public:
        using supper = ValuedPiece<Piece::BOOLEAN, char>;
        using self = BooleanPiece;

        BooleanPiece() : supper() {}

        BooleanPiece(const supper::Value &val) : supper(val) {}

        BooleanPiece(std::istream &bin) : supper(bin) {}

        const std::string str() const override {
            std::stringstream oss;
            oss << std::boolalpha << (get() != 0);
            return oss.str();
        }

        const std::string repr() const override {
            std::stringstream oss;
            oss << std::boolalpha << (get() != 0);
            return oss.str();
        }
    };

    class BinaryPiece : public TypedPiece<Piece::BINARY> {
    public:
        BinaryPiece() : m_buff() {}

        BinaryPiece(const std::string &buff) : m_buff(buff.data(), buff.size()) {}

        BinaryPiece(const binary &buff) : m_buff(buff) {}

        BinaryPiece(std::istream &bin) {
            this->read(bin);
        }

        BinaryPiece &set(const std::string &buff) {
            this->m_buff.clear();
            this->m_buff.write(buff.data(), buff.size());
            return *this;
        }

        BinaryPiece &set(const binary &buff) {
            this->m_buff = buff;
            return *this;
        }

        binary get() {
            return this->m_buff;
        }

        const binary get() const {
            return this->m_buff;
        }

        void clear() {
            this->m_buff.clear();
        }

        void dispose() {
            this->m_buff.dispose();
        }

        size_t size() const {
            return m_buff.size();
        }

        BinaryPiece &set_bits(const void *buffer, size_t size) {
            this->m_buff.clear();
            return push_bits(buffer, size);
        }

        BinaryPiece &push_bits(const void *buffer, size_t size) {
            this->m_buff.write(buffer, size);
            return *this;
        }

        template<typename T>
        BinaryPiece &push_bits(const T &val) {
            const void *_data = &val;
            size_t _size = sizeof(val);
            m_buff.write(_data, _size);
            return *this;
        }

        virtual std::istream &read(std::istream &bin) override {
            binio<binary>::read(bin, m_buff);
            return bin;
        }

        virtual std::ostream &write(std::ostream &bin) const override {
            binio<char>::write(bin, static_cast<char>(this->type()));
            binio<binary>::write(bin, m_buff);
            return bin;
        }

        const std::string str() const override {
            std::stringstream oss;
            oss << "\"@binary@" << size() << '\"';
            return oss.str();
        }

        const std::string repr() const override {
            std::stringstream oss;
            oss << "\"@binary@" << size() << '\"';
            return oss.str();
        }

    private:
        // std::ostringstream m_buff;
        binary m_buff;
    };

    class ListPiece : public TypedPiece<Piece::LIST> {
    public:
        ListPiece() {}

        ListPiece(size_t size) {
            m_list.resize(size);
        }

        ListPiece(std::istream &bin) {
            this->read(bin);
        }

        size_t size() const {
            return m_list.size();
        }

        void push_back(const std::shared_ptr<Piece> &pie) {
            m_list.push_back(pie);
        }

        void clear() {
            m_list.clear();
        }

        const std::shared_ptr<Piece> &index(size_t i) const {
            return m_list[i];
        }

        std::shared_ptr<Piece> &index(size_t i) {
            return m_list[i];
        }

        std::shared_ptr<Piece> &index(size_t i, const std::shared_ptr<Piece> &value) {
            return m_list[i] = value;
        }

        const std::shared_ptr<Piece> &operator[](size_t i) const {
            return m_list[i];
        }

        std::shared_ptr<Piece> &operator[](size_t i) {
            return m_list[i];
        }

        virtual std::istream &read(std::istream &bin) override {
            int size;
            binio<int>::read(bin, size);
            m_list.reserve(size);
            for (int i = 0; i < size; ++i) {
                m_list.push_back(Piece::Read(bin));
            }
            return bin;
        }

        virtual std::ostream &write(std::ostream &bin) const override {
            binio<char>::write(bin, static_cast<char>(this->type()));
            binio<int>::write(bin, static_cast<int>(this->size()));
            for (auto &pie : m_list) {
                /// TODO: check pointer pie valid
                pie->write(bin);
            }
            return bin;
        }

        const std::string str() const override {
            std::stringstream oss;
            oss << '[';
            for (size_t i = 0; i < size(); ++i) {
                /// TODO: check pointer pie valid
                if (i) oss << ", ";
                oss << index(i)->repr();
            }
            oss << ']';
            return oss.str();
        }

        const std::string repr() const override {
            std::stringstream oss;
            oss << '[';
            for (size_t i = 0; i < size(); ++i) {
                /// TODO: check pointer pie valid
                if (i) oss << ", ";
                oss << index(i)->repr();
            }
            oss << ']';
            return oss.str();
        }

    private:
        std::vector<std::shared_ptr<Piece>> m_list;
    };

    class DictPiece : public TypedPiece<Piece::DICT> {
    public:
        DictPiece() {}

        DictPiece(std::istream &bin) {
            this->read(bin);
        }

        size_t size() const {
            return m_dict.size();
        }

        size_t erase(const std::string &key) {
            return m_dict.erase(key);
        }

        bool has_key(const std::string &key) const {
            return m_dict.find(key) != m_dict.end();
        }

        std::shared_ptr<Piece> &index(const std::string &key) {
            return m_dict.at(key);
        }

        const std::shared_ptr<Piece> &index(const std::string &key) const {
            return m_dict.at(key);
        }

        std::shared_ptr<Piece> &index(const std::string &key, const std::shared_ptr<Piece> &value) {
            return m_dict[key] = value;
        }

        std::vector<std::string> keys() const {
            std::vector<std::string> result;
            result.reserve(m_dict.size());
            for (auto &key_pie : m_dict) {
                auto &key = key_pie.first;
                // auto &pie = key_pie.second;
                /// TODO: check pointer pie valid
                result.push_back(key);
            }
            return std::move(result);
        }

        template<size_t _size>
        std::shared_ptr<Piece> &operator[](const char (&key)[_size]) {
            return this->index(std::string(key));
        }

        template<size_t _size>
        const std::shared_ptr<Piece> &operator[](const char (&key)[_size]) const {
            return this->index(std::string(key));
        }

        std::shared_ptr<Piece> &operator[](const std::string &key) {
            return m_dict.at(key);
        }

        const std::shared_ptr<Piece> &operator[](const std::string &key) const {
            return m_dict.at(key);
        }

        virtual std::istream &read(std::istream &bin) override {
            int size;
            binio<int>::read(bin, size);
            m_dict.clear();
            for (int i = 0; i < size; ++i) {
                std::string key;
                binio<std::string>::read(bin, key);
                m_dict[key] = Piece::Read(bin);
            }
            return bin;
        }

        virtual std::ostream &write(std::ostream &bin) const override {
            binio<char>::write(bin, static_cast<char>(this->type()));
            binio<int>::write(bin, static_cast<int>(this->size()));
            for (auto &key_pie : m_dict) {
                auto &key = key_pie.first;
                auto &pie = key_pie.second;
                binio<std::string>::write(bin, key);
                /// TODO: check pointer pie valid
                pie->write(bin);
            }
            return bin;
        }

        const std::string str() const override {
            std::stringstream oss;
            oss << '{';
            bool first = true;
            for (auto &key : keys()) {
                /// TODO: check pointer pie valid
                if (first) first = false;
                else oss << ", ";
                oss << '\"' << key << "\": " << index(key)->repr();
            }
            oss << '}';
            return oss.str();
        }

        const std::string repr() const override {
            std::stringstream oss;
            oss << '{';
            bool first = true;
            for (auto &key : keys()) {
                /// TODO: check pointer pie valid
                if (first) first = false;
                else oss << ", ";
                oss << '\"' << key << "\": " << index(key)->repr();
            }
            oss << '}';
            return oss.str();
        }

    private:
        std::map<std::string, std::shared_ptr<Piece>> m_dict;
    };

    void Piece::Write(std::ostream &bin, const Piece &pie) {
        pie.write(bin);
    }

    void Piece::Write(std::ostream &bin, const std::shared_ptr<Piece> &pie) {
        pie.get()->write(bin);
    }

    std::shared_ptr<Piece> Piece::Read(std::istream &bin) {
        char type;
        binio<char>::read(bin, type);
        return Get(static_cast<Type>(type), bin);
    }

    std::shared_ptr<Piece> Piece::Get(Type type) {
        switch (static_cast<Type>(type)) {
            case NIL:
                return std::make_shared<NilPiece>();
            case INT:
                return std::make_shared<IntPiece>();
            case FLOAT:
                return std::make_shared<FloatPiece>();
            case STRING:
                return std::make_shared<StringPiece>();
            case BINARY:
                return std::make_shared<BinaryPiece>();
            case LIST:
                return std::make_shared<ListPiece>();
            case DICT:
                return std::make_shared<DictPiece>();
            case BOOLEAN:
                return std::make_shared<BooleanPiece>();
        }
        throw Exception("Unknown piece type.");
    }

    std::shared_ptr<Piece> Piece::Get(Type type, std::istream &bin) {
        auto pie = Get(type);
        pie->read(bin);
        return std::move(pie);
    }
}


#endif //ORZ_IO_JUG_PIECE_H
