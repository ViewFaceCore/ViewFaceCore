//
// Created by kier on 2020/2/3.
//

#ifndef TENNIS_UTILS_OTL_H
#define TENNIS_UTILS_OTL_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#include <string>
#include <sstream>

#include "except.h"
#include "log.h"

namespace ts {
    namespace otl {
        static inline unsigned int elf_hash(const char *str) {
            unsigned int hash = 0;
            unsigned int x = 0;
            while (*str) {
                hash = (hash << 4) + *str;
                if ((x = hash & 0xf0000000) != 0) {
                    hash ^= (x >> 24);
                    hash &= ~x;
                }
                str++;
            }
            return (hash & 0x7fffffff);
        }

        namespace sso {
            class OutOfRangeException : Exception {
            public:
                OutOfRangeException(size_t N, const std::string &std_string)
                        : Exception(Message(N, std_string)) {}

                static std::string Message(size_t N, const std::string &std_string) {
                    std::ostringstream oss;
                    oss << "Can not convert \"" << std_string << "\" (" << std_string.length() << ") to "
                        << "otl::sso::string<" << N << ">";
                    return oss.str();
                }
            };

            template<size_t N>
            class string {
            private:
                char m_buf[N] = {'\0'};
            public:
                using self = string;
                constexpr static size_t CAPACITY = N - 1;

                using iterator = char *;
                using const_iterator = const char *;

                string() {
                    // std::memset(m_buf, 0, N);
                }

                string(const string &other) = default;

                string(const std::string &std_string)
                        : self() {
                    if (std_string.length() > CAPACITY) {
                        TS_LOG_ERROR << OutOfRangeException::Message(N, std_string) << eject;
                    }
                    std::strcpy(m_buf, std_string.c_str());
                }

                string(const char *c_str)
                        : self() {
                    if (c_str == nullptr) return;
                    if (std::strlen(c_str) > CAPACITY) {
                        TS_LOG_ERROR << OutOfRangeException::Message(N, c_str) << eject;
                    }
                    std::strcpy(m_buf, c_str);
                }

                string(std::nullptr_t)
                        : self() {}

                const char *c_str() const { return m_buf; }

                size_t length() const {
                    return std::strlen(m_buf);
                }

                const char *data() const { return m_buf; }

                size_t size() const { return CAPACITY; }

                bool empty() const { return m_buf[0] == '\0'; }

                iterator begin() { return m_buf; }

                iterator end() { return m_buf + this->length(); }

                const_iterator begin() const { return m_buf; }

                const_iterator end() const { return m_buf + this->length(); }

                std::string std() const { return std::string(c_str()); }
            };

            template<size_t N1, size_t N2>
            inline bool operator==(const string<N1> &lhs, const string<N2> &rhs) {
                return std::strcmp(lhs.c_str(), rhs.c_str()) == 0;
            }

            template<size_t N1, size_t N2>
            inline bool operator!=(const string<N1> &lhs, const string<N2> &rhs) {
                return !operator==(lhs, rhs);
            }

            template<size_t N1, size_t N2>
            inline bool operator>(const string<N1> &lhs, const string<N2> &rhs) {
                return std::strcmp(lhs.c_str(), rhs.c_str()) > 0;
            }

            template<size_t N1, size_t N2>
            inline bool operator<(const string<N1> &lhs, const string<N2> &rhs) {
                return std::strcmp(lhs.c_str(), rhs.c_str()) < 0;
            }

            template<size_t N1, size_t N2>
            inline bool operator>=(const string<N1> &lhs, const string<N2> &rhs) {
                return !operator<(lhs, rhs);
            }

            template<size_t N1, size_t N2>
            inline bool operator<=(const string<N1> &lhs, const string<N2> &rhs) {
                return !operator>(lhs, rhs);
            }

            template<size_t N>
            inline std::ostream &operator<<(std::ostream &out, const string<N> &str) {
                return out << str.c_str();
            }

            template<>
            inline string<8>::string() {
                std::memset(m_buf, 0, 8);
            }

            template<>
            inline bool operator==(const string<8> &lhs, const string<8> &rhs) {
                return *reinterpret_cast<const uint64_t *>(lhs.data())
                       == *reinterpret_cast<const uint64_t *>(rhs.data());
            }

            template<>
            inline bool operator>(const string<8> &lhs, const string<8> &rhs) {
                return *reinterpret_cast<const uint64_t *>(lhs.data())
                       > *reinterpret_cast<const uint64_t *>(rhs.data());
            }

            template<>
            inline bool operator<(const string<8> &lhs, const string<8> &rhs) {
                return *reinterpret_cast<const uint64_t *>(lhs.data())
                       < *reinterpret_cast<const uint64_t *>(rhs.data());
            }
        }

        template <size_t N>
        using string = sso::string<N>;

        class VectorOutOfRangeException : Exception {
        public:
            VectorOutOfRangeException(size_t N, int i)
                    : Exception(Message(N, i)) {}

            static std::string Message(size_t N, int i) {
                std::ostringstream oss;
                oss << "Index " << i << " out of range of "<< "otl::vector<" << N << ">";
                return oss.str();
            }
        };

        template <typename T>
        class reverse_pointer {
        private:
            T *m_pointer = nullptr;
        public:
            using self = reverse_pointer;

            reverse_pointer() = default;

            reverse_pointer(T *pointer) : m_pointer(pointer) {}

            self operator++() { return --m_pointer; }

            self operator++(int) { return m_pointer--; }

            self operator--() { return ++m_pointer; }

            self operator--(int) { return m_pointer++; }

            self operator+(size_t offset) { return m_pointer - offset; }

            self operator-(size_t offset) { return m_pointer + offset; }

            bool operator==(const self &other) { return m_pointer == other.m_pointer; }

            bool operator!=(const self &other) { return !this->operator==(other); }

            T &operator*() { return *m_pointer; }

            T &operator*() const { return *m_pointer; }

            T *operator->() { return m_pointer; }

            T *operator->() const { return m_pointer; }
        };

        template <typename T>
        class const_reverse_pointer {
        private:
            const T *m_pointer = nullptr;
        public:
            using self = const_reverse_pointer;

            const_reverse_pointer() = default;

            const_reverse_pointer(const T *pointer) : m_pointer(pointer) {}

            self operator++() { return --m_pointer; }

            self operator++(int) { return m_pointer--; }

            self operator--() { return ++m_pointer; }

            self operator--(int) { return m_pointer++; }

            self operator+(size_t offset) { return m_pointer - offset; }

            self operator-(size_t offset) { return m_pointer + offset; }

            bool operator==(const self &other) { return m_pointer == other.m_pointer; }

            bool operator!=(const self &other) { return !this->operator==(other); }

            const T &operator*() { return *m_pointer; }

            const T &operator*() const { return *m_pointer; }

            const T *operator->() { return m_pointer; }

            const T *operator->() const { return m_pointer; }
        };

        template <typename T, size_t N, typename S = size_t>
        class vector {
        private:
            T m_buf[N];
            S m_size = 0;
        public:
            using self = vector;
            using value_type = T;
            using size_type = size_t;

            constexpr static size_t CAPACITY = N;

            using iterator = T *;
            using const_iterator = const T *;

            using reverse_iterator = reverse_pointer<T>;
            using const_reverse_iterator = const_reverse_pointer<T>;

            // copy
            explicit vector() = default;

            // from STL
            vector(const std::vector<T> &std_vector)
                : self(std_vector.begin(), std_vector.end()) {}

            // fill
            explicit vector (size_type n)
                : m_size(S(n)) {}

            vector(size_type n, const value_type& val)
                : m_size(S(n)) {
                for (auto &slot : *this) slot = val;
            }

            vector(int n, const value_type& val)
                    : m_size(S(n)) {
                for (auto &slot : *this) slot = val;
            }

            // range
            template <class InputIterator>
            vector(InputIterator first, InputIterator last) {
                size_type n = 0;
                for (auto it = first; it != last; ++it) {
                    if (n > CAPACITY) {
                        TS_LOG_ERROR << VectorOutOfRangeException::Message(N, n) << eject;
                    }
                    m_buf[n] = *it;
                    ++n;
                }
                m_size = S(n);
            }

            // copy
            vector(const vector& x) = default;

            // initializer list
            vector(std::initializer_list<value_type> il) {
                if (il.size() > CAPACITY) {
                    TS_LOG_ERROR << VectorOutOfRangeException::Message(N, il.size()) << eject;
                }
                size_type n = 0;
                for (auto it = il.begin(); it != il.end(); ++it) {
                    m_buf[n] = *it;
                    ++n;
                }
                m_size = S(n);
            }

            iterator begin() { return m_buf; }

            iterator end() { return m_buf + this->size(); }

            const_iterator begin() const { return m_buf; }

            const_iterator end() const { return m_buf + this->size(); }

            reverse_iterator rbegin() { return m_buf + this->size() - 1; }

            reverse_iterator rend() { return m_buf - 1; }

            const_reverse_iterator rbegin() const { return m_buf + this->size() - 1; }

            const_reverse_iterator rend() const { return m_buf - 1; }

            T *data() { return m_buf; }

            const T *data() const { return m_buf; }

            size_type size() const { return size_type(m_size); }

            size_type capacity() const { return CAPACITY; }

            std::vector<T> std() const { return std::vector<T>(begin(), end()); }

#define OTL_VECTOR_SUPPORT_INDEX(INDEX) \
            T &operator[](INDEX i) { return m_buf[i]; } \
            const T &operator[](INDEX i) const { return m_buf[i]; } \

            OTL_VECTOR_SUPPORT_INDEX(size_t)
            OTL_VECTOR_SUPPORT_INDEX(int32_t)
            OTL_VECTOR_SUPPORT_INDEX(int64_t)
#undef OTL_VECTOR_SUPPORT_INDEX

            bool empty() const { return m_size == 0; }

            T &front() { return m_buf[0]; }

            const T &front() const { return m_buf[0]; }

            T &back() { return m_buf[m_size - 1]; }

            const T &back() const { return m_buf[m_size - 1]; }

            void resize(size_type n, value_type val = value_type()) {
                *this = self(n, val);
            }

            iterator erase(iterator it) {
                this->_erase(_pointer2index(it), 1);
                return it;
            }

            iterator erase(iterator first, iterator last) {
                auto i = _pointer2index(first);
                auto j = _pointer2index(last);
                if (j < i) return first;
                this->_erase(i, j - i);
                return first;
            }

            iterator insert(iterator it, const T &val) {
                this->_insert(_pointer2index(it), val);
                return it;
            }

            template <class InputIterator>
            iterator insert(iterator it, InputIterator first, InputIterator last) {
                this->_insert(_pointer2index(it), self(first, last));
                return it;
            }

            void push_back(const T &val) {
                m_buf[m_size] = val;
                ++m_size;
            }

            void pop_back() {
                --m_size;
            }

            template <typename ...Args>
            void emplace_back(Args &&...args) {
                new(m_buf + m_size) T(std::forward<Args>(args)...);
                ++m_size;
            }

            void clear() {
                m_size = 0;
            }

            void reserve(size_type n) {}

        private:
            size_type _pointer2index(const T*ptr) const {
                return ptr < m_buf ? 0 : ptr - m_buf;
            }

            void _erase(size_type i, size_type len) {
                auto size = this->size();
                if (len + i > size) len = size - i;
                std::memmove(m_buf + i, m_buf + i + len, sizeof(T) * (this->size() - i - len));
                m_size -= len;
            }

            void _insert(size_type i, const T&val) {
                if (i + 1 > this->capacity()) {
                    TS_LOG_ERROR << VectorOutOfRangeException::Message(N, int(N)) << eject;
                }
                std::memmove(m_buf + i + 1, m_buf + i, sizeof(T) * (this->size() - i));
                m_buf[i] = val;
                m_size += 1;
            }

            template <size_t _N, typename _S>
            void _insert(size_type i, const vector<T, _N, _S> &vec) {
                if (i + vec.size() > this->capacity()) {
                    TS_LOG_ERROR << VectorOutOfRangeException::Message(N, int(N)) << eject;
                }
                std::memmove(m_buf + i + vec.size(), m_buf + i, sizeof(T) * (this->size() - i));
                std::memcpy(m_buf + i, vec.data(), sizeof(T) * (vec.size()));
                m_size += vec.size();
            }
        };

        template <typename T, size_t N, typename S>
        inline std::ostream &operator<<(std::ostream &out, const vector<T, N, S> &vec) {
            using size_type = typename vector<T, N, S>::size_type;
            std::ostringstream oss;
            oss << "[";
            for (size_type i = 0; i < vec.size(); ++i) {
                if (i) oss << ", ";
                oss << vec[i];
            }
            oss << "]";
            return out << oss.str();
        }

        template <typename T, size_t N1, typename S1, size_t N2, typename S2>
        inline bool operator==(const vector<T, N1, S1> &lhs, const vector<T, N2, S2> &rhs) {
            return lhs.size() == rhs.size() && std::memcmp(lhs.data(), rhs.data(), sizeof(T) * lhs.size()) == 0;
        }

        template <typename T, size_t N1, typename S1, size_t N2, typename S2>
        inline bool operator!=(const vector<T, N1, S1> &lhs, const vector<T, N2, S2> &rhs) {
            return !operator==(lhs, rhs);
        }


    }
}

namespace std {
    template<size_t N>
    struct hash<ts::otl::sso::string<N>> {
        using self = ts::otl::sso::string<N>;
        std::size_t operator()(const self &key) const {
            using std::size_t;
            using std::hash;

            return size_t(ts::otl::elf_hash(key.c_str()));
        }
    };
}

#endif //TENNIS_UTILS_OTL_H
