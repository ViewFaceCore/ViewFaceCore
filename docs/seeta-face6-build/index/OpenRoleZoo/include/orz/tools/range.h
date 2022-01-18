//
// Created by lby on 2018/3/22.
//

#ifndef ORZ_TOOLS_RANGE_H
#define ORZ_TOOLS_RANGE_H

#include <algorithm>
#include <cmath>
#include <cassert>
#include <vector>

namespace orz {

    template <typename T>
    class range {
    public:
        class iterator {
        public:
            explicit iterator(T value, T step_v)
                    : m_value(value), m_step_v(step_v) {
            }

            bool operator!=(const iterator &other) const {
                if (m_step_v >= 0)
                    return this->m_value < other.m_value;
                else
                    return this->m_value > other.m_value;
            }

            T operator*() const {
                return m_value;
            }

            iterator &operator++() {
                m_value += m_step_v;
                return *this;
            }

        private:
            T m_value;
            T m_step_v;
        };

        explicit range(T end_v)
                : m_begin_v(0), m_end_v(end_v), m_step_v(end_v >= T(0) ? T(1) : (T(-1) > 0 ? 1 : T(-1))) {
        }

        range(T begin_v, T end_v)
                : m_begin_v(begin_v), m_end_v(end_v), m_step_v(end_v >= begin_v ? T(1) : (T(-1) > 0 ? 1 : T(-1))) {
        }

        range(T begin_v, T end_v, T step_v)
                : m_begin_v(begin_v), m_end_v(end_v), m_step_v(step_v) {
        }

        iterator begin() const {
            return iterator(m_begin_v, m_step_v);
        }

        iterator end() const {
            return iterator(m_end_v, m_step_v);
        }

    private:
        T m_begin_v;
        T m_end_v;
        T m_step_v;
    };

    using irange = range<int>;
    using lrange = range<long>;
    using uirange = range<unsigned int>;
    using ulrange = range<unsigned long>;

    template <typename T>
    class binrange {
        class iterator {
        public:
            iterator(T value, T end_v, T bin_size)
                : m_value(value), m_end_v(end_v), m_bin_size(bin_size) {
            }

            bool operator!=(const iterator &other) const {
                return this->m_value < other.m_value;
            }

            range<T> operator*() const {
                return range<T>(m_value, std::min(m_end_v, m_value + m_bin_size));
            }

            iterator &operator++() {
                m_value += m_bin_size;
                return *this;
            }

        private:
            T m_value;
            T m_end_v;
            T m_bin_size;
        };

    public:
        binrange(T end_v, T clips)
                : m_begin_v(0), m_end_v(end_v), m_clips(clips) {
            assert(m_end_v >= m_begin_v);
        }

        binrange(T begin_v, T end_v, T clips)
                : m_begin_v(begin_v), m_end_v(end_v), m_clips(clips) {
            assert(m_end_v >= m_begin_v);
        }

        iterator begin() const {
            return iterator(m_begin_v, m_end_v,
                            static_cast<T>(std::ceil(static_cast<double>(m_end_v - m_begin_v) / m_clips)));
        }

        iterator end() const {
            return iterator(m_end_v, m_end_v,
                            static_cast<T>(std::ceil(static_cast<double>(m_end_v - m_begin_v) / m_clips)));
        }

    private:
        T m_begin_v;
        T m_end_v;
        T m_clips;
    };

    using ibinrange = binrange<int>;
    using lbinrange = binrange<long>;
    using uibinrange = binrange<unsigned int>;
    using ulbinrange = binrange<unsigned long>;


    /**
     * get `bins` bins split set [first, second)
     * @param first min number
     * @param second max number
     * @param bins number of bins
     * @return A list contains splited bins
     * @note Example input(0, 10, 3) returns [(0, 4), (4, 8), (8, 10)]
     */
    std::vector<std::pair<int, int>> split_bins(int first, int second, int bins);

    /**
     * get `bins` bins split set [first, second)
     * @param first min number
     * @param second max number
     * @param bins number of bins
     * @return A list contains splited bins
     * @note Example input(0, 10, 3) returns [(0, 4), (4, 8), (8, 10)]
     */
    std::vector<std::pair<size_t, size_t>> lsplit_bins(size_t first, size_t second, size_t bins);
}


#endif //ORZ_TOOLS_RANGE_H
