//
// Created by Lby on 2017/8/12.
//

#include "pot.h"
#include <cstring>

namespace ts {

    static std::shared_ptr<void> default_allocator(size_t _size)
    {
        return std::shared_ptr<void>(std::malloc(_size), std::free);
    }

    Pot::Pot()
        : Pot(default_allocator) {
    }

    Pot::Pot(const allocator &ator)
        :  m_allocator(ator == nullptr ? default_allocator : ator), m_capacity(0), m_size(0), m_data() {
    }

    void *Pot::malloc(size_t _size) {
        if (_size > m_capacity) {
            m_data = m_allocator(_size);
            m_capacity = _size;
        }
        m_size = _size;
        return m_data.get();
    }

    void *Pot::relloc(size_t _size) {
        if (_size > m_capacity) {
            auto new_data = m_allocator(_size);
            std::memcpy(new_data.get(), m_data.get(), m_capacity);
            m_data = new_data;
            m_capacity = _size;
        }
        m_size = _size;
        return m_data.get();
    }

    void *Pot::data() const {
        return m_data.get();
    }

    size_t Pot::capacity() const {
        return m_capacity;
    }

    size_t Pot::size() const {
        return m_size;
    }

    void Pot::dispose() {
        m_capacity = 0;
        m_data.reset();
    }
}
