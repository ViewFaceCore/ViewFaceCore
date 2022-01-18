//
// Created by kier on 2019/2/21.
//

#ifndef TENSORSTACK_CORE_TENSOR_ITERATOR_H
#define TENSORSTACK_CORE_TENSOR_ITERATOR_H

#include <vector>
#include "utils/except.h"

#include <utils/api.h>

namespace ts {
    class TS_DEBUG_API ShapeIterator {
    public:
        using self = ShapeIterator;
        template <typename T>
        using Shape = otl::vector<T, 7, T>;

        explicit ShapeIterator(const Shape<int> &shape)
                : m_shape(shape) {
            m_coordinate.resize(m_shape.size(), 0);
        }

        ShapeIterator(const Shape<int> &shape, const Shape<int> &coordinate)
                : m_shape(shape), m_coordinate(coordinate) {
        }

        self &operator++() {
            increase();
            return *this;
        }

        const self operator++(int) {
            auto temp = self(m_shape, m_coordinate);
            increase();
            return temp;
        }

        self &operator--() {
            decrease();
            return *this;
        }

        const self operator--(int) {
            auto temp = self(m_shape, m_coordinate);
            decrease();
            return temp;
        }

        void rewind() {
            for (auto &i : m_coordinate) {
                i = 0;
            }
        }

        const Shape<int> &coordinate() {
            return m_coordinate;
        }

    private:
        void increase() {
            increase(int(m_coordinate.size() - 1));
        }

        void decrease() {
            decrease(int(m_coordinate.size() - 1));
        }

        void increase(int dim) {
            while (dim >= 0) {
                if (m_coordinate[dim] + 1 == m_shape[dim]) {
                    m_coordinate[dim] = 0;
                    --dim;
                    continue;
                } else {
                    m_coordinate[dim]++;
                    break;
                }
            }
        }

        void decrease(int dim) {
            while (dim >= 0) {
                if (m_coordinate[dim] - 1 < 0) {
                    m_coordinate[dim] = m_shape[dim] - 1;
                    --dim;
                    continue;
                } else {
                    m_coordinate[dim]--;
                    break;
                }
            }
        }


        Shape<int> m_shape;
        Shape<int> m_coordinate;

    public:
        ShapeIterator(const self &other) = default;
        ShapeIterator &operator=(const self &other) = default;

        ShapeIterator(self &&other) {
            *this = std::move(other);
        }
        ShapeIterator &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_shape);
            MOVE_MEMBER(m_coordinate);
#undef MOVE_MEMBER
            return *this;
        }
    };

    class TS_DEBUG_API HypeShape {
    public:
        using self = HypeShape;
        using T = int32_t;

        template <typename T>
        using Shape = otl::vector<T, 7, T>;

        explicit HypeShape(const Shape<int32_t> &shape)
                : m_shape(shape) {
            // update weights
            if (m_shape.empty()) return;
            m_weights.resize(m_shape.size());
            auto size = m_shape.size();
            auto weight_it = m_weights.rbegin();
            auto shape_it = m_shape.rbegin();
            *weight_it++ = *shape_it++;
            for (size_t times = size - 1; times; --times) {
                *weight_it = *(weight_it - 1) * *shape_it;
                ++weight_it;
                ++shape_it;
            }
        }

        T to_index(const std::initializer_list<T> &coordinate) const {
            // if (coordinate.size() > m_shape.size()) throw CoordinateOutOfShapeException(m_shape, coordinate);
            if (coordinate.size() == 0) return 0;
            auto size = coordinate.size();
            auto weight_it = m_weights.end() - size + 1;
            auto coordinate_it = coordinate.begin();
            T index = 0;
            for (size_t times = size - 1; times; --times) {
                index += *weight_it * *coordinate_it;
                ++weight_it;
                ++coordinate_it;
            }
            index += *coordinate_it;
            return index;
        }

        T to_index(const std::vector<T> &coordinate) const {
            // if (coordinate.size() > m_shape.size()) throw CoordinateOutOfShapeException(m_shape, coordinate);
            if (coordinate.empty()) return 0;
            auto size = coordinate.size();
            auto weight_it = m_weights.end() - size + 1;
            auto coordinate_it = coordinate.begin();
            T index = 0;
            for (size_t times = size - 1; times; --times) {
                index += *weight_it * *coordinate_it;
                ++weight_it;
                ++coordinate_it;
            }
            index += *coordinate_it;
            return index;
        }

        T to_index(const Shape<T> &coordinate) const {
            // if (coordinate.size() > m_shape.size()) throw CoordinateOutOfShapeException(m_shape, coordinate);
            if (coordinate.empty()) return 0;
            auto size = coordinate.size();
            auto weight_it = m_weights.end() - size + 1;
            auto coordinate_it = coordinate.begin();
            T index = 0;
            for (size_t times = size - 1; times; --times) {
                index += *weight_it * *coordinate_it;
                ++weight_it;
                ++coordinate_it;
            }
            index += *coordinate_it;
            return index;
        }

        T to_index(int arg0) {
            return arg0;
        }

#define LOOP_HEAD(n) constexpr size_t size = (n); auto weight_it = m_weights.end() - size + 1; T index = 0;
#define LOOP_ON(i) index += *weight_it * arg##i; ++weight_it;
#define LOOP_END(i) index += arg##i; return index;

        T to_index(int arg0, int arg1) {
            LOOP_HEAD(2)
            LOOP_ON(0)
            LOOP_END(1)
        }

        T to_index(int arg0, int arg1, int arg2) {
            LOOP_HEAD(3)
            LOOP_ON(0) LOOP_ON(1)
            LOOP_END(2)
        }

        T to_index(int arg0, int arg1, int arg2, int arg3) {
            LOOP_HEAD(4)
            LOOP_ON(0) LOOP_ON(1) LOOP_ON(2)
            LOOP_END(3)
        }

        T to_index(int arg0, int arg1, int arg2, int arg3, int arg4) {
            LOOP_HEAD(5)
            LOOP_ON(0) LOOP_ON(1) LOOP_ON(2) LOOP_ON(3)
            LOOP_END(4)
        }

        T to_index(int arg0, int arg1, int arg2, int arg3, int arg4,
                   int arg5) {
            LOOP_HEAD(6)
            LOOP_ON(0) LOOP_ON(1) LOOP_ON(2) LOOP_ON(3) LOOP_ON(4)
            LOOP_END(5)
        }

        T to_index(int arg0, int arg1, int arg2, int arg3, int arg4,
                   int arg5, int arg6) {
            LOOP_HEAD(7)
            LOOP_ON(0) LOOP_ON(1) LOOP_ON(2) LOOP_ON(3) LOOP_ON(4)
            LOOP_ON(5)
            LOOP_END(6)
        }

        T to_index(int arg0, int arg1, int arg2, int arg3, int arg4,
                   int arg5, int arg6, int arg7) {
            LOOP_HEAD(8)
            LOOP_ON(0) LOOP_ON(1) LOOP_ON(2) LOOP_ON(3) LOOP_ON(4)
            LOOP_ON(5) LOOP_ON(6)
            LOOP_END(7)
        }

        T to_index(int arg0, int arg1, int arg2, int arg3, int arg4,
                   int arg5, int arg6, int arg7, int arg8) {
            LOOP_HEAD(9)
            LOOP_ON(0) LOOP_ON(1) LOOP_ON(2) LOOP_ON(3) LOOP_ON(4)
            LOOP_ON(5) LOOP_ON(6) LOOP_ON(7)
            LOOP_END(8)
        }

        T to_index(int arg0, int arg1, int arg2, int arg3, int arg4,
                   int arg5, int arg6, int arg7, int arg8, int arg9) {
            LOOP_HEAD(10)
            LOOP_ON(0) LOOP_ON(1) LOOP_ON(2) LOOP_ON(3) LOOP_ON(4)
            LOOP_ON(5) LOOP_ON(6) LOOP_ON(7) LOOP_ON(8)
            LOOP_END(9)
        }

#undef LOOP_HEAD
#undef LOOP_ON
#undef LOOP_END

        std::vector<T> to_coordinate(T index) const {
            // if (m_shape.empty()) return std::vector<T>();
            // if (index >= m_weights[0]) throw IndexOutOfShapeException(m_shape, index);
            if (m_shape.empty())
                return std::vector<T>();
            std::vector<T> coordinate(m_shape.size());
            to_coordinate(index, coordinate);
            return std::move(coordinate);
        }

        void to_coordinate(T index, std::vector<T> &coordinate) const {
            if (m_shape.empty()) {
                coordinate.clear();
                return;
            }
            coordinate.resize(m_shape.size());
            auto size = m_shape.size();
            auto weight_it = m_weights.begin() + 1;
            auto coordinate_it = coordinate.begin();
            for (size_t times = size - 1; times; --times) {
                *coordinate_it = index / *weight_it;
                index %= *weight_it;
                ++weight_it;
                ++coordinate_it;
            }
            *coordinate_it = index;
        }

        T count() const { return m_weights.empty() ? 1 : m_weights[0]; }

        T weight(size_t i) const { return m_weights[i]; };

        const Shape<T> &weight() const { return m_weights; };

        T shape(size_t i) const { return m_shape[i]; };

        const Shape<T> &shape() const { return m_shape; };

        explicit operator Shape<int>() const { return m_shape; }

    private:
        Shape<int32_t> m_shape;
        Shape<T> m_weights;

    public:
        HypeShape(const self &other) = default;
        HypeShape &operator=(const self &other) = default;

        HypeShape(self &&other) {
            *this = std::move(other);
        }
        HypeShape &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_shape);
            MOVE_MEMBER(m_weights);
#undef MOVE_MEMBER
            return *this;
        }
    };
}

#endif //TENSORSTACK_TENSOR_ITERATOR_H
