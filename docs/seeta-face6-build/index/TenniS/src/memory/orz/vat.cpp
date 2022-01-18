//
// Created by Lby on 2017/8/23.
//

#include "utils/except.h"
#include "vat.h"
#include <algorithm>
#include <utils/log.h>

namespace ts {

    Vat::Vat()
        : self(nullptr) {
    }

    Vat::Vat(const Pot::allocator &ator)
        : m_allocator(ator) {
        // TS_LOG_DEBUG << "new Vat() -> " << this;
    }

    Vat::~Vat() {
        // TS_LOG_DEBUG << "delete Vat(" << this << ")";
    }

    static int binary_find(const std::vector<Pot> &heap, size_t size, int left, int right) {
        while (true) {
            if (right - left <= 0) return left;
            int middle = (left + right) / 2;
            if (heap[middle].capacity() < size) {
                left = middle + 1;
                continue;
            }
            if (middle > 0 && heap[middle - 1].capacity() >= size) {
                right = middle - 1;
                continue;
            }
            return middle;
        }
    }

    // find the position of Pot, which can contain the size, or return 0
    // the return value
    static int binary_find(const std::vector<Pot> &heap, size_t size) {
        return binary_find(heap, size, 0, int(heap.size()) - 1);
    }

    void *Vat::malloc(size_t _size) {
        if (_size == 0) return nullptr;
        // find first small piece
        Pot pot(m_allocator);
        if (!m_heap.empty())
        {
            auto i = binary_find(m_heap, _size);
            pot = m_heap[i];
            m_heap.erase(m_heap.begin() + i);
        }
        void *ptr = pot.malloc(_size);
        m_dict.insert(std::pair<void *, Pot>(ptr, pot));

        return ptr;
    }

    void Vat::free(const void *ptr) {
        if (ptr == nullptr) return;
        auto key = const_cast<void *>(ptr);
        auto it = m_dict.find(key);
        if (it == m_dict.end()) {
            throw Exception("Can not free this ptr");
        }

        if (!m_deprecated) {
            auto &pot = it->second;
            auto i = binary_find(m_heap, pot.capacity());
            auto ind = m_heap.begin() + i;
            m_heap.insert(ind, pot);
        }

        m_dict.erase(it);
    }

    void Vat::reset() {
        for (auto &pair : m_dict) {
            m_heap.push_back(pair.second);
        }
        m_dict.clear();
        std::sort(m_heap.begin(), m_heap.end(), [](const Pot &p1, const Pot &p2){return p1.capacity() < p2.capacity(); });
    }

    void Vat::dispose() {
        m_dict.clear();
        m_heap.clear();
    }

    void Vat::swap(Vat &that)
    {
        this->m_heap.swap(that.m_heap);
        this->m_dict.swap(that.m_dict);
    }

    Vat::Vat(Vat &&that)
    {
        this->swap(that);
    }

    Vat &Vat::operator=(Vat &&that)
    {
        this->swap(that);
        return *this;
    }

    void Vat::clean() {
        this->m_heap.clear();
        this->m_heap.shrink_to_fit();
    }

    void Vat::deprecated() {
        this->clean();
        m_deprecated = true;
    }

    uint64_t Vat::summary() const {
        uint64_t sum = 0;
        for (auto &a : this->m_heap) {
            sum += a.capacity();
        }
        for (auto &b : this->m_dict) {
            sum  += b.second.capacity();
        }
        return sum;
    }
}
