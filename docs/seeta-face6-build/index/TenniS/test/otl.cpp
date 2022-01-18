//
// Created by kier on 2020/2/3.
//

#include "utils/otl.h"

int main() {
    ts::otl::sso::string<8> a = "1234567";
    a = "13";
    std::cout << a << std::endl;

    ts::otl::vector<int32_t, 15> b = {1, 2, 3};
    std::cout << b << std::endl;
}

