//
// Created by lby on 2018/3/29.
//

#include <orz/tools/range.h>

#include <iostream>

int main() {
    for (auto bin : orz::ibinrange(0, 111, 10)) {
        for (auto i : bin) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
}