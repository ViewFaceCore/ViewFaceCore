//
// Created by kier on 2018/6/3.
//

#ifndef TENSORSTACK_CORE_SCAN_H
#define TENSORSTACK_CORE_SCAN_H

#include <vector>
#include <cstdint>
#include <climits>
#include <memory>
#include <iostream>
#include <utility>
#include <stack>

namespace ts {
    class  Iteration {
    public:
        using step_t = long;    ///< step type
        static const step_t outer = LONG_MAX;   ///< means step into outer area
        static const step_t finish = LONG_MIN;  ///< means iteration was over
        using times_t = size_t;
        // using count_t = times_t; // no counter in this version

        enum Type {
            STEP,
            ITERS
        };

        explicit Iteration(Type type) : type(type) {}

        Iteration(Type type, times_t times) : type(type), times(times) {}

        virtual ~Iteration() {};

        Type type;
        times_t times = 0;
        // count_t count = 0;
    };

    template<Iteration::Type _type>
    class IterationBase : public Iteration {
    public:
        IterationBase() : Iteration(_type) {}

        IterationBase(times_t times) : Iteration(_type, times) {}
    };

    class IterationStep : public IterationBase<Iteration::STEP> {
    public:
        using supper = IterationBase<Iteration::STEP>;

        IterationStep() = default;

        explicit IterationStep(times_t times) : supper(times) {}

        explicit IterationStep(times_t times, step_t step) : supper(times), step(step) {}

        step_t step = 0;
    };

    class IterationIters : public IterationBase<Iteration::ITERS> {
    public:
        using supper = IterationBase<Iteration::ITERS>;

        IterationIters() = default;

        explicit IterationIters(times_t times) : supper(times) {}

        explicit IterationIters(times_t times, const std::vector<Iteration *> &iters) : supper(times), iters(iters) {}

        explicit IterationIters(times_t times, std::vector<Iteration *> &&iters) : supper(times),
                                                                                   iters(std::move(iters)) {}

        std::vector<Iteration *> iters; // those pointers should allocate by new
    };

    inline IterationStep *new_iteration_step() {
        return new IterationStep();
    }

    inline IterationIters *new_iteration_iters() {
        return new IterationIters();
    }

    inline IterationStep *new_iteration_step(Iteration::times_t times) {
        return new IterationStep(times);
    }

    inline IterationIters *new_iteration_iters(Iteration::times_t times) {
        return new IterationIters(times);
    }

    inline Iteration *new_iteration(Iteration::times_t times, Iteration::step_t step) {
        return new IterationStep(times, step);
    }

    inline Iteration *new_iteration(Iteration::times_t times, Iteration *iters) {
        return new IterationIters(times, {iters});
    }

    inline Iteration *new_iteration(Iteration::times_t times, const std::vector<Iteration *> &iters) {
        return new IterationIters(times, iters);
    }

    inline Iteration *new_iteration(Iteration::times_t times, std::vector<Iteration *> &&iters) {
        return new IterationIters(times, std::move(iters));
    }

    inline void delete_iteration(const Iteration *iteration) {
        if (iteration == nullptr) return;
        if (iteration->type == Iteration::ITERS) {
            auto iters = reinterpret_cast<const IterationIters *>(iteration)->iters;
            for (auto *iter : iters) delete_iteration(iter);
        }
        delete iteration;
    }

    inline Iteration *clone_iteration(const Iteration *iteration) {
        if (iteration == nullptr) return nullptr;
        if (iteration->type == Iteration::ITERS) {
            using ThisIteration = IterationIters;
            auto iteration_iters = reinterpret_cast<const ThisIteration *>(iteration);
            std::unique_ptr<IterationIters, decltype(&delete_iteration)> iteration_clone(new_iteration_iters(),
                                                                                         &delete_iteration);
            iteration_clone->times = iteration_iters->times;
            iteration_clone->iters.resize(iteration_iters->iters.size(), nullptr);
            auto from = iteration_iters->iters.begin();
            auto to = iteration_clone->iters.begin();
            while (from != iteration_iters->iters.end()) {
                *to = clone_iteration(*from);
                ++from;
                ++to;
            }
            return iteration_clone.release();
        } else {
            using ThisIteration = IterationStep;
            auto iteration_step = reinterpret_cast<const ThisIteration *>(iteration);
            return new_iteration(iteration_step->times, iteration_step->step);
        }
    }

    // return all iteration may size, and remove all zero times iteration
    // if return zero, iteration should be deleted
    Iteration::times_t update_iteration(Iteration *iteration) {
        if (iteration == nullptr) return 0;
        if (iteration->times == 0) {
            delete_iteration(iteration);
            return 0;
        }
        if (iteration->type == Iteration::ITERS) {
            Iteration::times_t times = 0;
            using ThisIteration = IterationIters;
            auto iteration_iters = reinterpret_cast<ThisIteration *>(iteration);
            auto iter = iteration_iters->iters.begin();
            while (iter != iteration_iters->iters.end()) {
                Iteration::times_t local_times = update_iteration(*iter);
                if (local_times == 0) {
                    iter = iteration_iters->iters.erase(iter);
                } else {
                    times += local_times;
                    ++iter;
                }
            }
            if (times == 0) delete_iteration(iteration);
            return iteration->times * times;
        }
        return iteration->times;
    }

    Iteration::times_t update_iteration(Iteration **piteration) {
        if (piteration == nullptr) return 0;
        Iteration::times_t times = update_iteration(*piteration);
        if (times == 0) *piteration = nullptr;
        return times;
    }

    class IterationDescriptor {
    public:
        using self = IterationDescriptor;

        static const auto outer = Iteration::outer;
        static const auto finish = Iteration::finish;
        using times_t = Iteration::times_t;
        using step_t = Iteration::step_t;

        class Group {
        public:
            template<typename Arg0>
            static void try_insert_group_rape(std::vector<Iteration *> &group, Arg0 &&arg0) {
                group.push_back(try_rape(std::forward<Arg0>(arg0)));
            };

            template<typename Arg0, typename... Args>
            static void try_insert_group_rape(std::vector<Iteration *> &group, Arg0 &&arg0, Args &&... args) {
                group.push_back(try_rape(std::forward<Arg0>(arg0)));
                try_insert_group_rape(group, std::forward<Args>(args)...);
            };

            template<typename... Args>
            static std::vector<Iteration *> try_return_group_rape(Args &&... args) {
                std::vector<Iteration *> group;
                try_insert_group_rape(group, std::forward<Args>(args)...);
                return std::move(group);
            };

            template<typename... Args>
            Group(Args &&... args) : m_proto_group(try_return_group_rape(std::forward<Args>(args)...)) {}

            ~Group() {
                for (auto &iter : m_proto_group) delete_iteration(iter);
            }

        private:
            std::vector<Iteration *> m_proto_group;

            friend class IterationDescriptor;
        };

        IterationDescriptor(times_t times, step_t step) : m_proto(new_iteration(times, step)) {}

        IterationDescriptor(times_t times, const IterationDescriptor &descriptor)
                : m_proto(new_iteration(times, try_rape(descriptor))) {}

        IterationDescriptor(times_t times, IterationDescriptor &&descriptor)
                : m_proto(new_iteration(times, try_rape(std::move(descriptor)))) {}

        IterationDescriptor(times_t times, const Group &group) : m_proto(
                new_iteration(times, try_rape(group))) {}

        IterationDescriptor(times_t times, Group &&group) : m_proto(
                new_iteration(times, try_rape(std::move(group)))) {}

        IterationDescriptor(const IterationDescriptor &descriptor) : m_proto(try_rape(descriptor)) {}

        IterationDescriptor(IterationDescriptor &&descriptor) : m_proto(try_rape(std::move(descriptor))) {}

        ~IterationDescriptor() { delete_iteration(m_proto); }

        self &operator=(const IterationDescriptor &descriptor) {
            auto new_iteration = clone_iteration(descriptor.m_proto);
            std::swap(m_proto, new_iteration);
            delete_iteration(new_iteration);
            return *this;
        };

        self &operator=(IterationDescriptor &&descriptor) {
            this->swap(descriptor);
            return *this;
        }

        IterationDescriptor clone() const {
            IterationDescriptor doly;
            doly.m_proto = clone_iteration(this->m_proto);
            return std::move(doly);
        }

        const Iteration *proto() const { return m_proto; }

        Iteration *proto() { return m_proto; }

        times_t update() { return update_iteration(&m_proto); }

    private:
        IterationDescriptor() = default;

        static Iteration *reap(IterationDescriptor &descriptor) {
            Iteration *raw_iter = nullptr;
            std::swap(raw_iter, descriptor.m_proto);
            return raw_iter;
        }

        static Iteration *try_rape(const IterationDescriptor &descriptor) {
            return clone_iteration(descriptor.m_proto);
        }

        static Iteration *try_rape(IterationDescriptor &&descriptor) {
            return reap(descriptor);
        }

        static std::vector<Iteration *> try_rape(const Group &group) {
            std::vector<Iteration *> raw_iters = group.m_proto_group;
            for (auto &iter : raw_iters) iter = clone_iteration(iter);
            return std::move(raw_iters);
        }

        static std::vector<Iteration *> try_rape(Group &&group) {
            std::vector<Iteration *> raw_iters;
            std::swap(raw_iters, group.m_proto_group);
            return std::move(raw_iters);
        }

        void swap(IterationDescriptor &descriptor) {
            std::swap(m_proto, descriptor.m_proto);
        }

        Iteration *m_proto = nullptr;
    };

    class IterationInterpreter {
    public:
        static const auto outer = Iteration::outer;
        static const auto finish = Iteration::finish;
        using times_t = Iteration::times_t;
        using step_t = Iteration::step_t;
        using count_t = Iteration::times_t;

        class Status {
        public:
            const Iteration *proto;
            count_t count;

            explicit Status(const Iteration *proto) : proto(proto), count(proto->times) {}

            bool finished() const { return count == 0; }

            void next() { --count; }

            Iteration::Type type() const { return proto->type; }

            Iteration::step_t step() const { return reinterpret_cast<const IterationStep *>(proto)->step; }

            const std::vector<Iteration *>
            iters() const { return reinterpret_cast<const IterationIters *>(proto)->iters; }
        };

        void bind(Iteration *proto) {
            this->m_proto = proto;
            rewind();
        }

        void bind(IterationDescriptor &descriptor) {
            this->m_proto = descriptor.proto();
            rewind();
        }

        void rewind() {
            m_status = decltype(m_status)();
            if (m_proto == nullptr) return;
            m_status.emplace(m_proto);
            deploy();
        }

        // set status to situation of after next and before step
        void deploy() {
            while (!m_status.empty()) {
                auto &top = m_status.top();
                if (top.finished()) {
                    m_status.pop();
                    continue;
                }
                if (top.type() == Iteration::ITERS) {
                    top.next();
                    auto iters = top.iters();
                    auto it = iters.rbegin();
                    while (it != iters.rend()) {
                        m_status.emplace(*it);
                        ++it;
                    }
                    continue;
                }
                break;
            }
        }

        // set iteration to next, may call deploy, return just step
        step_t next() {
            if (m_status.empty()) return finish;
            auto &top = m_status.top();
            auto step = top.step();
            top.next();
            deploy();
            return step;
        }

        // get this time step
        step_t step() const {
            if (m_status.empty()) return finish;
            auto &top = m_status.top();
            auto step = top.step();
            return step;
        }

    private:
        const Iteration *m_proto = nullptr;
        std::stack<Status> m_status;
    };

    using Scan = IterationDescriptor;
    using Loop = IterationInterpreter;

}


#endif //TENSORSTACK_CORE_SCAN_H
