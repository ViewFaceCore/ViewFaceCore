//
// Created by lby on 2018/1/11.
//

#include <orz/codec/base64.h>
#include <orz/utils/log.h>
#include <cstdlib>

bool base64_test(const std::string &bin) {
    std::string codes = orz::base64_encode(bin);
    std::string decode_bin = orz::base64_decode(codes);
    ORZ_LOG(orz::DEBUG) << bin << " vs. " << decode_bin;
    return decode_bin == bin;
}

const std::string random_string() {
    int size = std::rand() % 1024;
    std::string bin(size, 0);
    for (auto &ch : bin) {
        ch = char(std::rand() % 256);
    }
    return std::move(bin);
}

int main(int argc, char *argv[]) {
    std::srand(7726);
    int N = 32768;
    int count = 0;
    for (int i = 0; i < N; ++i) {
        std::string bin = random_string();
        if (base64_test(bin)) ++count;
    }

    ORZ_LOG(orz::INFO) << "Test count: " << N << ", Succeed count: " << count << ".";

    return 0;
}