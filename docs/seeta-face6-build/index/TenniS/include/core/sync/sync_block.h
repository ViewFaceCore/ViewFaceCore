//
// Created by kier on 2018/12/26.
//

#ifndef TENSORSTACK_SYNC_SYNC_BLOCK_H
#define TENSORSTACK_SYNC_SYNC_BLOCK_H

#include <functional>
#include <map>
#include <utility>

#include <utils/mutex.h>
#include <utils/log.h>

#include <utils/api.h>

namespace ts {
    template <typename _KEY, typename _VALUE>
    class TS_DEBUG_API SyncBlock {
    public:
        using self = SyncBlock;
        using shared = std::shared_ptr<self>;

        using key_t = _KEY;
        using value_t = _VALUE;

        using sync_handler = std::function<_VALUE(const _VALUE &from_value, const _KEY &from_key, const _KEY &to_key)>;

        SyncBlock(const self &) = delete;
        self &operator=(const self &) = delete;

        SyncBlock(const _KEY &key, const _VALUE &value, const sync_handler &handler, bool need_lock)
                : m_param(Param::Shared(handler)) {
            m_default_key = key;
            if (need_lock) m_mutex = std::make_shared<ts::rwmutex>();
            this->set(key, value);
        }

    private:
        void set(const _KEY &key, const _VALUE &value) {
            auto _write = this->lock_write();
            if (key == m_default_key) {
                m_param->m_sync_values.clear();
                auto pair_it_flag = m_param->m_sync_values.insert(std::make_pair(key, value));
                m_default_value = &(pair_it_flag.first->second);
            } else {
                m_param->m_sync_values.clear();
                m_param->m_sync_values.insert(std::make_pair(key, value));
                auto default_key = key;
                auto default_value = m_param->m_hanlder(value, key, m_default_key);
                auto pair_it_flag = m_param->m_sync_values.insert(std::make_pair(default_key, default_value));
                m_default_value = &(pair_it_flag.first->second);
            }
        }

        _VALUE &get(const _KEY &key) {
            auto _read = this->lock_read();
            if (key == m_default_key) return *m_default_value;
            auto it = m_param->m_sync_values.find(key);
            if (it == m_param->m_sync_values.end()) {
                TS_LOG_ERROR << "Can not access key=" << key << eject;
            }
            return it->second;
        }

        const _VALUE &get(const _KEY &key) const {
            return const_cast<self *>(this)->get(key);
        }

        void clear() {
            auto _write = this->lock_write();
            auto kept_value = *m_default_key;
            m_param->m_sync_values.clear();
            auto pair_it_flag = m_param->m_sync_values.insert(std::make_pair(m_default_key, kept_value));
            m_default_value = &(pair_it_flag.first->second);
        }

        void clear(const _KEY &key) {
            auto _write = this->lock_write();
            if (key == m_default_key) TS_LOG_ERROR << "Can not clear default key=" << key << eject;
            m_param->m_sync_values.erase(key);
        }

    public:
        _VALUE &sync(const _KEY &key) {
            {
                auto _read = this->lock_read();
                if (key == m_default_key) return *m_default_value;
                auto it = m_param->m_sync_values.find(key);
                if (it != m_param->m_sync_values.end()) {
                    return it->second;
                }
            }
            {
                auto _write = this->lock_write();
                return this->sync_insert(key);
            }
        }

        const _KEY &key() const {
            return m_default_key;
        }

        const _VALUE &value() const {
            auto _read = this->lock_read();
            return *m_default_value;
        }

        shared view(const _KEY &key) {
            std::shared_ptr<self> dolly(new self);
            if (key == m_default_key) {
                auto _write = this->lock_read();
                dolly->m_default_key = m_default_key;
                dolly->m_default_value = m_default_value;
                dolly->m_param = m_param;
                dolly->m_mutex = m_mutex;
            } else {
                auto _write = this->lock_write();
                _VALUE *dolly_value = nullptr;

                auto it = m_param->m_sync_values.find(key);
                if (it != m_param->m_sync_values.end()) {
                    dolly_value = &it->second;
                } else {
                    dolly_value = &this->sync_insert(key);
                }

                dolly->m_default_key = key;
                dolly->m_default_value = dolly_value;
                dolly->m_param = m_param;
                dolly->m_mutex = m_mutex;
            }
            return std::move(dolly);
        }

        // broadcast default value to other key,
        // be careful, this action may disable already exist view
        void broadcast() {
            auto _write = this->lock_write();
            auto key = m_default_key;
            auto value = *m_default_value;
            m_param->m_sync_values.clear();
            auto pair_it_flag = m_param->m_sync_values.insert(std::make_pair(key, value));
            m_default_value = &(pair_it_flag.first->second);
        }

        void foreach(const std::function<void(const _KEY &key, const _VALUE &value)> &handler) const {
            auto _read = this->lock_read();
            for (auto &pair : this->m_param->m_sync_values) {
                handler(pair.first, pair.second);
            }
        }

    private:
        SyncBlock() = default;

        _VALUE &sync_insert(const _KEY &key) {
            if (key == m_default_key) return *m_default_value;
            auto it = m_param->m_sync_values.find(key);
            if (it != m_param->m_sync_values.end()) {
                return it->second;
            }
            _VALUE value = m_param->m_hanlder(*m_default_value, m_default_key, key);
            auto pair_it = m_param->m_sync_values.insert(std::make_pair(key, value));
            return pair_it.first->second;
        }

        using unique_read_lock = ts::unique_read_lock<ts::rwmutex>;
        using unique_write_lock = ts::unique_write_lock<ts::rwmutex>;

        std::unique_ptr<unique_read_lock> lock_read() const {
            if (m_mutex) {
                return std::unique_ptr<unique_read_lock>(new unique_read_lock(*m_mutex));
            } else {
                return std::unique_ptr<unique_read_lock>(nullptr);
            }
        }

        std::unique_ptr<unique_write_lock> lock_write() const {
            if (m_mutex) {
                return std::unique_ptr<unique_write_lock>(new unique_write_lock(*m_mutex));
            } else {
                return std::unique_ptr<unique_write_lock>(nullptr);
            }
        }

        class Param {
        public:
            using self = Param;
            using shared = std::shared_ptr<self>;

            std::map<_KEY, _VALUE> m_sync_values;
            sync_handler m_hanlder;

            Param(const sync_handler &handler)
                    : m_hanlder(handler) {
            }

            static shared Shared(const sync_handler &handler) {
                return std::make_shared<self>(handler);
            }
        };

        _KEY m_default_key;
        _VALUE *m_default_value;

        std::shared_ptr<Param> m_param;
        std::shared_ptr<ts::rwmutex> m_mutex;

        // std::shared_ptr<unique_write_lock> m_locked;
    public:
        SyncBlock(self &&other) {
            *this = std::move(other);
        }
        SyncBlock &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_default_key);
            MOVE_MEMBER(m_default_value);
            MOVE_MEMBER(m_param);
            MOVE_MEMBER(m_mutex);
#undef MOVE_MEMBER
            return *this;
        }
    };
}

#endif //TENSORSTACK_SYNC_SYNC_BLOCK_H
