//
// Created by lby on 2018/1/10.
//

#ifndef ORZ_CODEC_JSON_ITERATOR_H
#define ORZ_CODEC_JSON_ITERATOR_H

#include "orz/utils/log.h"

namespace orz {

    class json_iterator {
    public:
        using self = json_iterator;

        json_iterator(const char *data, int size, int index = 0)
                : data(data), size(size), index(index) {
        }

        const char &operator*() const {
            if (index < 0 || index >= size) ORZ_LOG(ERROR) << "index out of range" << crash;
            return data[index];
        }

        const json_iterator begin() const {
            return json_iterator(data, size, 0);
        }

        const json_iterator end() const {
            return json_iterator(data, size, size);
        }

        json_iterator &operator++() {
            ++index;
            return *this;
        }

        const json_iterator operator++(int) {
            return json_iterator(data, size, index++);
        }

        json_iterator &operator+=(int shift) {
            index += shift;
            return *this;
        }

        json_iterator &operator-=(int shift) {
            index -= shift;
            return *this;
        }

        bool operator==(const json_iterator &it) const {
            return self::data == it.data && self::size == it.size && self::index == it.index;
        }

        bool operator!=(const json_iterator &it) const {
            return !self::operator==(it);
        }

        friend const json_iterator operator+(const json_iterator &it, int shift) {
            return json_iterator(it.data, it.size, it.index + shift);
        }

        friend const json_iterator operator+(int shift, const json_iterator &it) {
            return json_iterator(it.data, it.size, it.index + shift);
        }

        friend const json_iterator operator-(const json_iterator &it, int shift) {
            return json_iterator(it.data, it.size, it.index - shift);
        }

        friend int operator-(const json_iterator &lhs, const json_iterator &rhs) {
            if (lhs.data != rhs.data) ORZ_LOG(ERROR) << "can not sub iterators from different init" << crash;
            return lhs.index - rhs.index;
        }

        const std::string cut(json_iterator end) const {
            int length = end - *this;
            int over = end - this->end();
            if (over > 0) length -= over;
            if (length <= 0) return std::string();
            return std::string(data + index, length);
        }

    private:
        const char *data;
        int size;
        int index;
    };
}

#endif //ORZ_CODEC_JSON_ITERATOR_H
