//
// Created by kier on 2019/1/28.
//

#ifndef TENSORSTACK_BOARD_BOARD_H
#define TENSORSTACK_BOARD_BOARD_H


#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <numeric>
#include <string>
#include <algorithm>

#include <utils/api.h>

namespace ts {
    template <typename T>
    class TS_DEBUG_API Statistics {
    public:
        using self = Statistics;
        using Datum = T;

    private:
        std::vector<Datum> m_data;
        using iterator = typename std::vector<Datum>::iterator;
        using const_iterator = typename std::vector<Datum>::const_iterator;

    public:
        void append(const Datum datum) {
            this->m_data.push_back(datum);
        }

        void clear() {
            this->m_data.clear();
        }

        Datum max() const {
            if (this->m_data.empty()) return Datum(0);
            return std::minmax(this->m_data.begin(), this->m_data.end()).second;
        }

        Datum min() const {
            if (this->m_data.empty()) return Datum(0);
            return std::minmax(this->m_data.begin(), this->m_data.end()).first;
        }

        Datum avg() const {
            return this->m_data.empty() ? Datum(0) : (this->sum() / this->count());
        }

        Datum sum() const {
            return std::accumulate(this->m_data.begin(), this->m_data.end(), Datum(0));
        }

        size_t count() const {
            return this->m_data.size();
        }

        iterator begin() {
            return this->m_data.begin();
        }

        iterator end() {
            return this->m_data.end();
        }

        const_iterator begin() const {
            return this->m_data.begin();
        }

        const_iterator end() const {
            return this->m_data.end();
        }
    };

    template <typename T>
    class TS_DEBUG_API Board {
    public:
        using self = Board;
        using Datum = typename Statistics<T>::Datum;
        using key_type = std::string;
        using value_type = Statistics<T>;
        using pair = std::pair<key_type, value_type>;

    private:
        std::unordered_map<key_type, value_type> m_board;
        using iterator = typename std::unordered_map<key_type, value_type>::iterator;
        using const_iterator = typename std::unordered_map<key_type, value_type>::const_iterator;

    public:
        void append(const key_type &name, const Datum datum) {
            this->m_board[name].append(datum);
        }

        void clear(const key_type &name) {
            this->query(name).clear();
        }

        Datum max(const key_type &name) const {
            return this->query(name).max();
        }

        Datum min(const key_type &name) const {
            return this->query(name).min();
        }

        Datum avg(const key_type &name) const {
            return this->query(name).avg();
        }

        Datum sum(const key_type &name) const {
            return this->query(name).sum();
        }

        size_t count(const key_type &name) const {
            return this->query(name).count();
        }

        void clean() {
            this->m_board.clear();
        }

        iterator begin() {
            return this->m_board.begin();
        }

        iterator end() {
            return this->m_board.end();
        }

        const_iterator begin() const {
            return this->m_board.begin();
        }

        const_iterator end() const {
            return this->m_board.end();
        }

        iterator find(const key_type &name) {
            return this->m_board.find(name);
        }

        const_iterator find(const key_type &name) const {
            return this->m_board.find(name);
        }

        value_type query(const key_type &name) const {
            auto it = this->m_board.find(name);
            if (it == this->m_board.end()) return Statistics<Datum>();
            return it->second;
        }
    };
}


#endif //TENSORSTACK_BOARD_BOARD_H
