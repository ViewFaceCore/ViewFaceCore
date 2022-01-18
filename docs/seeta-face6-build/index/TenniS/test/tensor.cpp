//
// Created by seeta on 2018/5/25.
//

#include <core/tensor.h>
#include <core/dtype.h>
#include <iostream>
#include <core/scan.h>
#include <global/setup.h>

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
    out << "(";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i) out << ", ";
        out << vec[i];
    }
    out << ")";
    return out;
}

int main() {
    using namespace ts;
    setup();

    Tensor a(FLOAT32, {2, 3, 4});


    float *data = a.data<float>();
    for (int i = 0; i < a.count(); ++i) {
        data[i] = i;
    }

//    while (true)
//    {
//        if (it.data() == it.end()) break;
//
//        std::cout << *it.data<float>() << std::endl;
//
//        it.next();
//    }

    auto b = Scan(3, 1);
    auto c = Scan(3, -3);
    auto d = Scan(3, -3);
    auto e = Scan(2, c);
    Scan::Group g(Scan(0, 12), Scan(0, 1));
    auto gg = Scan(3, std::move(g));
    auto f = Scan(1, {Scan(1, 0), std::move(b), Scan(3, 2), std::move(gg), std::move(e)});
    std::cout << std::endl;

    std::cout << f.update() << std::endl;

    auto t = f.clone();
    std::cout << std::endl;

    std::cout << std::endl;

    std::cout << std::endl;

    Loop loop;
    loop.bind(t);

    while (true) {
        auto step = loop.next();
        if (step == Loop::finish) break;
        std::cout << step << " ";
    }

    std::cout << std::endl;

    std::vector<int> shape = {2, 3, 4, 5};
    std::vector<int> new_dims = {0, 3, 1, 2};

    HypeShape hype_shape(shape);

    std::cout << "count: " << hype_shape.count() << std::endl;

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    std::cout << hype_shape.to_index({i, j, k, l}) << std::endl;
                }
            }
        }
    }

    std::cout << std::endl;
    std::cout << hype_shape.to_index({1, 2340}) << std::endl;

    return 0;
}

