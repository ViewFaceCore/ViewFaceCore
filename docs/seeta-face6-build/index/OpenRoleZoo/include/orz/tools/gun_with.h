//
// Created by seeta on 2018/7/30.
//

#ifndef ORZ_TOOLS_GUN_WITH_H
#define ORZ_TOOLS_GUN_WITH_H

#include "multi.h"
#include "orz/sync/shotgun.h"

namespace orz {
    template<typename T>
    class GunWithMulti : public Multi<T> {
    public:
        using self = GunWithMulti;
        using supper = Multi<T>;

        using Mission = std::function<void(T*)>;

        template<typename... Args>
        explicit GunWithMulti(size_t N, Args &&... args)
                : supper(N, std::forward<Args>(args)...), m_gun(N) {
        }

        void fire(const Mission &mission) {
            this->m_gun.fire([this, mission](int id) { mission(this->core(id)); });
        }

        template<typename FUNC, typename... Args>
        void bind(FUNC func, Args &&... args) {
            this->m_gun.fire(this->bind_core(func, std::forward<Args>(args)...));
        }

        void join() {
            this->m_gun.join();
        }

    private:
        template<typename FUNC, typename... Args>
        void fire_core(int id, FUNC func, Args &&... args) {
            std::bind(func, &this->core(id), std::forward<Args>(args)...)();
        }

        template<typename FUNC, typename... Args>
        orz::Cartridge::bullet_type bind_core(FUNC func, Args &&... args) {
            // no errors here
            return std::bind(&self::fire_core<FUNC, Args...>, this, std::placeholders::_1, func, std::forward<Args>(args)...);
        }

        orz::Shotgun m_gun;
    };
}

#endif //ORZ_TOOLS_GUN_WITH_H
