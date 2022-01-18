//
// Created by seeta on 2018/7/31.
//

#include <orz/tools/multi.h>
#include <chrono>
#include <iostream>


class B
{
public:
    mutable int b = 0;
public:
    void add() { b++; }
    void add_const() const { b++; }
};

void add(B *b)
{
    b->add();
}

void add_const(const B *b)
{
    b->add_const();
}

void add_ref(B &b)
{
    b.add();
}

void add_const_ref(const B &b)
{
    b.add_const();
}

int main()
{
    using namespace orz;
    int N = 10000;
    Multi<B> mb(10000);
    using namespace std::chrono;
    microseconds duration(0);
    auto start = system_clock::now();
    auto end = system_clock::now();
    duration += duration_cast<microseconds>(end - start);
    double spent = 1.0 * duration.count() / 1000;
    // std::cout << "Full takes " << spent << " ms " << std::endl;

    start = system_clock::now();
    for (int i = 0; i < N; ++i) mb.each(&B::add);
    end = system_clock::now();
    duration = duration_cast<microseconds>(end - start);
    spent = 1.0 * duration.count() / 1000;
    std::cout << "mb.each(&B::add); Full takes " << spent << " ms " << std::endl;

    start = system_clock::now();
    for (int i = 0; i < N; ++i) mb.each(add);
    end = system_clock::now();
    duration = duration_cast<microseconds>(end - start);
    spent = 1.0 * duration.count() / 1000;
    std::cout << "mb.each(&add); Full takes " << spent << " ms " << std::endl;

    mb.each(&add);
    mb.each(&add_const);
    // mb.each(&add_ref);
    // mb.each(&add_const_ref);
    return 0;
}
