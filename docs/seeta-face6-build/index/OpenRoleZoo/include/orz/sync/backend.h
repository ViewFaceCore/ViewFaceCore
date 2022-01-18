//
// Created by lby on 2018/6/5.
//

#ifndef ORZ_SYNC_BACKEND_H
#define ORZ_SYNC_BACKEND_H

#include "cartridge.h"

namespace orz {
    template<typename T>
    class Backend {
    public:
        using self = Backend;
        using Mission = std::function<T()>;

        Backend()
                : m_busy(false) {
        }

        explicit Backend(const T &value)
                : m_busy(false), m_future(value) {
        }

        T fire(const Mission &mission) {
            this->put(mission);
            return this->get();
        }

        template <typename FUNC, typename... Args>
        T bind(FUNC func, Args &&... args) {
            return this->fire(std::bind(func, std::forward<Args>(args)...));
        };

        bool busy() const { return this->m_busy; }

        void put(const Mission &mission) {
            if (!this->m_busy) {
                this->m_busy = true;
                this->m_cart.fire(0, [this, mission](int) {
                    T a = mission();
                    std::unique_lock<std::mutex> _locker(this->m_mutex);
                    this->m_future = std::move(a);
                }, [this](int) {
                    this->m_busy = false;
                });
            }
        }

        T get() const {
            std::unique_lock<std::mutex> _locker(this->m_mutex);
            return this->m_future;
        }

        void set(const T &value) {
            std::unique_lock<std::mutex> _locker(this->m_mutex);
            this->m_future = value;
        }

    private:
        mutable std::mutex m_mutex;
        T m_future;
        orz::Cartridge m_cart;
        mutable std::atomic<bool> m_busy;
    };
}


#endif //ORZ_SYNC_BACKEND_H
