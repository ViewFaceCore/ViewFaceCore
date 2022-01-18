//
// Created by Lby on 2017/10/19.
//

#ifndef ORZ_TOOLS_LIMIT_H
#define ORZ_TOOLS_LIMIT_H

#include <atomic>
#include <stdexcept>
#include <string>

namespace orz
{
    template <int N, typename T>
    class limit {
    public:
        limit() {
            ++_instance_count;
            if (_instance_count > N) throw std::logic_error(
                        std::string("Can not create more the ") +
                        std::to_string(N) +
                        std::string(" instances.")
                );
        }
        virtual ~limit() {
            --_instance_count;
        }
    private:
        static std::atomic<int> _instance_count;
    };

    template <int N, typename T>
    std::atomic<int> limit<N, T>::_instance_count;
}

#endif //ORZ_TOOLS_LIMIT_H
