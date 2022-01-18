//
// Created by Lby on 2017/10/31.
//

#include "orz/io/jug/binary.h"

#include <cstring>
#include <algorithm>

namespace orz {

    binary::binary() {}

    binary::binary(size_t _size) {
        resize(_size);
    }

    binary::binary(const void *_buffer, size_t _size) {
        write(_buffer, _size);
    }

//    binary::binary(const std::string &str) {
//        write(str.data(), str.size());
//    }

//    binary::operator std::string() const {
//        return std::string(self::data<char>(), self::size());
//    }

    size_t binary::size() const { return m_size; }

    size_t binary::capacity() const { return m_capacity; }

    size_t binary::read(void *_buffer, size_t _size) const {
        size_t memory_left = m_size - m_index;
        size_t can_read = std::min<size_t>(memory_left, _size);
        std::memcpy(_buffer, now_data(), can_read);
        m_index += can_read;
        return can_read;
    }

    size_t binary::write(const void *_buffer, size_t _size) {
        size_t memory_right = m_index + _size;
        reverse(memory_right);
        std::memcpy(now_data(), _buffer, _size);
        if (memory_right > m_size) m_size = memory_right;
        m_index += _size;
        return _size;
    }

    size_t binary::get_pos() const {
        return m_index;
    }

    size_t binary::set_pos(pos _pos, int _shift) {
        m_index = correct_index(_pos, _shift);
        return m_index;
    }

    void binary::shift(int _size) {
        set_pos(pos::now, _size);
    }

    const void *binary::data() const {
        return m_data.get();
    }

    void *binary::data() {
        return m_data.get();
    }

    binary binary::clone() const {
        binary doly;
        doly.write(self::data(), self::size());
        return std::move(doly);
    }

    void binary::memset(char ch) {
        std::memset(self::data(), ch, self::capacity());
    }

    void binary::memset(pos _pos, int _begin, int _end, char ch) {
        size_t c_begin = correct_index(_pos, _begin);
        size_t c_end = correct_index(_pos, _end);
        size_t c_size = c_end - c_begin;
        std::memset(self::data<char>() + c_begin, ch, c_size);
    }

    void binary::reverse(size_t _size) {
        if (_size > m_capacity) {
            auto *new_data = std::malloc(_size);
            std::memcpy(new_data, self::data(), self::size());
            m_data.reset(new_data, std::free);
            m_capacity = _size;
        }
    }

    void binary::resize(size_t _size) {
        reverse(_size);
        m_size = _size;
    }

    void binary::clear() {
        m_index = 0;
        m_size = 0;
    }

    void binary::dispose() {
        m_index = 0;
        m_size = 0;
        m_capacity = 0;
        m_data.reset();
    }

    size_t binary::correct_index(int _index) {
        int c_index = std::max<int>(0, std::min<int>(static_cast<int>(m_size), _index));
        return static_cast<size_t>(c_index);
    }

    size_t binary::correct_index(pos _pos, int _shift) {
        size_t _base = m_index;
        switch (_pos) {
            case pos::beg:
                _base = 0;
                break;
            case pos::now:
                _base = m_index;
                break;
            case pos::end:
                _base = m_size;
                break;
        }
        int b_index = static_cast<int>(_base) + _shift;
        return correct_index(b_index);
    }

    void *binary::now_data() { return self::data<char>() + m_index; }

    const void *binary::now_data() const { return const_cast<self *>(this)->now_data(); }

    bool binary::empty() const {
        return m_size == 0;
    }

    std::string to_string(const binary &str) {
        return std::string(str.data<char>(), str.size());
    }

    binary to_binary(const std::string &bin) {
        return binary(bin.data(), bin.size());
    }

    bool operator==(const binary &lhs, const binary &rhs) {
        if (lhs.size() != rhs.size()) return false;
        using long_step_type = size_t;
        size_t size = lhs.size();
        size_t long_step = sizeof(long_step_type);
        size_t long_step_size = size / long_step;
        for (size_t i = 0;i < long_step_size; ++i) {
            if (lhs.data<long_step_type>()[i] != rhs.data<long_step_type>()[i]) return false;
        }
        for (size_t i = long_step_size * long_step; i < size; ++i) {
            if (lhs.data<unsigned char>()[i] != rhs.data<unsigned char>()[i]) return false;
        }
        return true;
    }

    bool operator!=(const binary &lhs, const binary &rhs) {
        return !operator==(lhs, rhs);
    }
}