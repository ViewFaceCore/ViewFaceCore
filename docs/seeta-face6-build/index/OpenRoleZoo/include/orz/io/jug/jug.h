//
// Created by Lby on 2017/8/16.
//

#ifndef ORZ_IO_JUG_JUG_H
#define ORZ_IO_JUG_JUG_H

#include "piece.h"
#include <cstddef>

namespace orz {

    static const int STA_MASK = 0x19910929;

    /*
     * @brief interface of piece
     */
    class jug {
    public:
        using self = jug;

        jug();

        jug(Piece::Type type);

        jug(std::nullptr_t _);

        jug(int val);

        jug(unsigned int val) : jug(static_cast<int>(val)) {}

        jug(char val) : jug(static_cast<int>(val)) {}

        jug(unsigned char val) : jug(static_cast<int>(val)) {}

        jug(short val) : jug(static_cast<int>(val)) {}

        jug(unsigned short val) : jug(static_cast<int>(val)) {}

        jug(long val) : jug(static_cast<int>(val)) {}

        jug(unsigned long val) : jug(static_cast<int>(val)) {}

        jug(long long val) : jug(static_cast<int>(val)) {}

        jug(unsigned long long val) : jug(static_cast<int>(val)) {}

        jug(float val);

        jug(double val) : jug(static_cast<float>(val)) {}

        jug(const std::string &val);

        template<size_t _size>
        jug(const char (&val)[_size]) : jug(std::string(val)) {}

        jug(const char *val) : jug(std::string(val)) {}

        jug(bool val);

        jug(const binary &val);

        bool valid(Piece::Type type) const;

        bool valid() const;

        bool invalid(Piece::Type type) const {return !this->valid(type);}

        bool invalid() const {return !this->valid();}

        // common type set function
        jug &operator=(std::nullptr_t _);

        jug &operator=(int val);

        jug &operator=(unsigned int val) { return *this = static_cast<int>(val); }

        jug &operator=(char val) { return *this = static_cast<int>(val); }

        jug &operator=(unsigned char val) { return *this = static_cast<int>(val); }

        jug &operator=(short val) { return *this = static_cast<int>(val); }

        jug &operator=(unsigned short val) { return *this = static_cast<int>(val); }

        jug &operator=(long val) { return *this = static_cast<int>(val); }

        jug &operator=(unsigned long val) { return *this = static_cast<int>(val); }

        jug &operator=(long long val) { return *this = static_cast<int>(val); }

        jug &operator=(unsigned long long val) { return *this = static_cast<int>(val); }

        jug &operator=(float val);

        jug &operator=(double val) { return *this = static_cast<float>(val); }

        jug &operator=(const std::string &val);

        template<size_t _size>
        jug &operator=(const char (&val)[_size]) { return *this = std::string(val); }

        jug &operator=(const char *val) { return *this = std::string(val); }

        jug &operator=(const binary &val);

        jug &operator=(bool val);

        // common type function
        bool to_bool() const;

        int to_int() const;

        float to_float() const;

        std::string to_string() const;

        binary to_binary() const;

        // common type function
        operator bool() const;

        operator int() const;

        operator float() const;

        operator std::string() const;

        operator binary() const;

        // string, binary, list, dict function
        size_t size() const;

        // list function
        jug index(size_t i);

        const jug index(size_t i) const;

        jug operator[](size_t i) { return index(i); }

        const jug operator[](size_t i) const { return index(i); }

        jug operator[](int i) { return index(i); }

        const jug operator[](int i) const { return index(i); }

        jug index(size_t i, const jug &value);

        jug &append(const jug &value);

        // dict function
        jug index(const std::string &key);

        const jug index(const std::string &key) const;

        jug operator[](const std::string &key) { return index(key); }

        const jug operator[](const std::string &key) const { return index(key); }

        template<size_t _size>
        jug operator[](const char(&key)[_size]) { return index(key); }

        template<size_t _size>
        const jug operator[](const char(&key)[_size]) const { return index(key); }

        jug operator[](const char *key) { return index(key); }

        const jug operator[](const char *key) const { return index(key); }

        jug index(const std::string &key, const jug &value);

        std::vector<std::string> keys() const;

        // binary function
        jug &set_bits(const void *buffer, size_t size);

        jug &push_bits(const void *buffer, size_t size);

        template<typename T>
        jug &push_bits(const T &val) {
            switch (m_pie->type()) {
                case Piece::BINARY: {
                    auto buff = reinterpret_cast<BinaryPiece *>(m_pie.get());
                    buff->push_bits<T>(val);
                    return *this;
                }
                default:
                    throw Exception("This jug has no method push_bits<T>(val)");
            }
        }

        Piece *raw();

        const Piece *raw() const;

        const std::string str() const;

        const std::string repr() const;

    private:
        jug(const std::shared_ptr<Piece> &pie) : m_pie(pie) {}

        operator std::shared_ptr<Piece>() const { return m_pie; }

        std::shared_ptr<Piece> m_pie;

    public:
        friend std::ostream &operator<<(std::ostream &out, const jug &e);

        friend jug jug_parse(const std::string &buffer);

        friend std::string jug_build(const jug &j);

        friend jug jug_read(const std::string &filename);

        friend jug jug_read(std::istream &in);

        friend bool jug_write(const std::string &filename, const jug &j);

        friend void jug_write(std::ostream &out, const jug &j);

        friend jug sta_read(const std::string &filename, int mask);

        friend jug sta_read(std::istream &in, int mask);

        friend bool sta_write(const std::string &filename, const jug &j, int mask);

        friend void sta_write(std::ostream &out, const jug &j, int mask);
    };

    std::ostream &operator<<(std::ostream &out, const jug &e);

    jug jug_parse(const std::string &buffer);

    std::string jug_build(const jug &j);

    jug jug_read(const std::string &filename);

    jug jug_read(std::istream &in);

    bool jug_write(const std::string &filename, const jug &j);

    void jug_write(std::ostream &out, const jug &j);

    jug sta_read(const std::string &filename, int mask = STA_MASK);

    jug sta_read(std::istream &in, int mask = STA_MASK);

    bool sta_write(const std::string &filename, const jug &j, int mask = STA_MASK);

    void sta_write(std::ostream &out, const jug &j, int mask = STA_MASK);

    template<typename T>
    T jug_get(const jug &j, const T &def = T()) {
        try {
            return static_cast<T>(j);
        } catch (const Exception &) {
            return def;
        }
    }

    std::string to_string(const jug& obj);
}


#endif //ORZ_IO_JUG_JUG_H
