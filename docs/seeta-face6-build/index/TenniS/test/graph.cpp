//
// Created by kier on 2018/10/16.
//

#include <module/graph.h>
#include <iostream>

template <typename T>
class Add {
public:
    T forward(const std::vector<ts::Node> &inputs) {
        T sum = 0;
        for (auto &input : inputs) {
            // sum += input.ref<int>();
        }
        return sum;
    }
};

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const Add<T> &node) {
    return out << "Add";
}

int main()
{
    int N= 1000;
    for (int i = 0; i < N; ++i) {
        ts::Graph g;
        auto a = g.make("a");
        auto b = g.make("b");
        auto c = g.make("c");
        ts::Node::Link(c, {a, b});
    }
}
