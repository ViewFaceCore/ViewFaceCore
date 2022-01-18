//
// Created by Lby on 2017/9/30.
//

#ifndef ORZ_LEGO_BRICK_H
#define ORZ_LEGO_BRICK_H

#include <functional>
#include <memory>
#include <vector>
#include <deque>
#include "orz/sync/shotgun.h"
#include "orz/utils/log.h"

namespace orz {

    template <typename T>
    using IOList = std::vector<T>;

    template <typename T>
    using IOQueue = std::deque<T>;  // for sliding window

    template<typename I, typename O>
    class brick {
    public:
        using ptr = std::shared_ptr<brick<I, O>>;
        using IType = I;
        using OType = O;

        virtual OType work(const IType &input) = 0;
        OType operator()(const IType &input) {return this->work(input);}
    };

    template<typename I, typename O>
    class function_brick : public brick<I, O> {
    public:
        using step_type = std::function<O(const I &)>;

        template<typename FUNC>
        function_brick(const FUNC &func) {
            this->m_step = [func](const I &input) { return func(input); };
        }

        virtual O work(const I &input) override {
            return this->m_step(input);
        }

    private:
        step_type m_step;
    };

    template <typename I, typename O>
    using borrow_brick = function_brick<I, O>;

    template<typename I, typename T, typename O>
    class big_brick : public brick<I, O> {
    public:
        using step1_type = typename brick<I, T>::ptr;
        using step2_type = typename brick<T, O>::ptr;

        big_brick(step1_type step1, step2_type step2)
                : m_step1(step1), m_step2(step2) {}

        virtual O work(const I &input) override {
            return this->m_step2->work(this->m_step1->work(input));
        }

    private:
        step1_type m_step1;
        step2_type m_step2;
    };

    template <typename I, typename O>
    class wide_brick : public brick<IOList<I>, IOList<O>> {
    public:
        wide_brick(typename brick<I, O>::ptr core)
                : m_gun(0) {
            m_core.push_back(core);
        }

        wide_brick(const std::vector<typename brick<I, O>::ptr> &cores)
                : m_gun(cores.size()), m_core(cores) {
            if (m_core.empty()) orz::Log(FATAL) << "Can not init with empty cores" << crash;
        }

        virtual IOList<O> work(const IOList<I> &input) override
        {
            IOList<O> output(input.size());
            for (size_t i = 0; i < input.size(); ++i)
            {
                m_gun.fire([i, &input, &output, this](int id){
                    output[i] = this->m_core[id]->work(input[i]);
                });
            }
            m_gun.join();
            return std::move(output);
        }
    private:
        Shotgun m_gun;
        std::vector<typename brick<I, O>::ptr> m_core;
    };

    template <typename BRICK, typename... Args>
    typename brick<typename BRICK::IType, typename BRICK::OType>::ptr make_brick(Args&&... args) {
        return std::make_shared<BRICK>(std::forward<Args>(args)...);
    };

    template <typename BRICK, typename... Args>
    typename brick<IOList<typename BRICK::IType>, IOList<typename BRICK::OType>>::ptr make_wide_brick(const size_t N, Args&&... args) {
        using I = typename BRICK::IType;
        using O = typename BRICK::OType;
        if (N == 1)
        {
            auto core = make_brick<BRICK>(std::forward<Args>(args)...);
            return make_brick<wide_brick<I, O>>(core);
        }
        else
        {
            std::vector<typename brick<I, O>::ptr> cores(N);
            for (auto &core : cores) core = make_brick<BRICK>(std::forward<Args>(args)...);
            return make_brick<wide_brick<I, O>>(cores);
        }
    };


}

#endif //ORZ_BRICK_H
