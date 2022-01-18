//
// Created by lby on 2018/1/16.
//

#include <orz/ssl/rsa.h>
#include <orz/utils/log.h>
#include <cstdlib>

std::string mem_pub_key =
        "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCzJf/ehpbp5qK4FJ3YISCe0KrM\n"
        "QiyIzqibJZGsF1PQxPmzyzJkewFTCcEECzXqwrv9Q1x7Cu40b7tGxPcS+fBnqucn\n"
        "G8dosKnCw6+8r7J1AaKHB/7EUngaA4nVBbuHMgXEv9cE5ENfSPKUPo9MbOIx206L\n"
        "P7UyQwse+jB40TIwRQIDAQAB";
std::string mem_key =
        "MIICXgIBAAKBgQCzJf/ehpbp5qK4FJ3YISCe0KrMQiyIzqibJZGsF1PQxPmzyzJk\n"
        "ewFTCcEECzXqwrv9Q1x7Cu40b7tGxPcS+fBnqucnG8dosKnCw6+8r7J1AaKHB/7E\n"
        "UngaA4nVBbuHMgXEv9cE5ENfSPKUPo9MbOIx206LP7UyQwse+jB40TIwRQIDAQAB\n"
        "AoGBAIfQQw5cUoS4iJutZYy4cJZ180Yu1LxSj5gu/yTL+orHCda4MVfjuLlPJ7j9\n"
        "Fr8HKqVyL+ZH/xZZrkyUfgsw2IM6ZAQV9dtvhQp+Vv3mg4+ULR1WY+2e0M7Ly3Rd\n"
        "2pn5NVXjERB9i142MBkUc4KIHW+148dShxq2KhtRGYBshCaBAkEA1j22O3ZIb6SO\n"
        "iqQALbmsdn7DD2H9HNkK08Kw1i5xclhWFGzqXtkAkhSud7fCF0vgpu9Pw3C9GqQj\n"
        "324PnLn4DQJBANYROK639sdwHqBgxeXx07HstSzvciMEs2fzKf24iEKje3Ik2vyl\n"
        "hc3QkQZG/DqBZhDNI/+apLL7PlGUF4QgExkCQQCJvV1nN3H0zVCTlENFIqXd/Tu9\n"
        "rRtFq8lJQlfdLDjl8iNNuISqfEvgn4lYEP2parBBw4R9vALomPUzVhiVg/8VAkEA\n"
        "vTrZ/XD3wFM3b5Q8PElqVlUlzwQXxCbqpNBKZVPTd/zmKf+0aTO0tTxShtcHLnym\n"
        "ieGdmdzNDL6V1y1vIEfuOQJAGJC2daFroBdRaZ+OhRE1SzNSUr/0Y7Ag1sIvm5/A\n"
        "umVUY+zCtnKIMmi4srM4OvcHqjZsf3R7kF08/IRuaAq0DA==";

bool rsa_test1(const std::string &bin) {
    std::string codes = orz::rsa_private_encode("orz.key", bin);
    std::string decode_bin = orz::rsa_public_decode("orz_pub.key", codes);
    ORZ_LOG(orz::DEBUG) << bin << " vs. " << decode_bin;
    return decode_bin == bin;
}

bool rsa_test2(const std::string &bin) {
    std::string codes = orz::rsa_public_encode("orz_pub.key", bin);
    std::string decode_bin = orz::rsa_private_decode("orz.key", codes);
    ORZ_LOG(orz::DEBUG) << bin << " vs. " << decode_bin;
    return decode_bin == bin;
}

bool rsa_test3(const std::string &bin) {
    std::string codes = orz::rsa_mem_private_encode(mem_key, bin);
    std::string decode_bin = orz::rsa_mem_public_decode(mem_pub_key, codes);
    ORZ_LOG(orz::DEBUG) << bin << " vs. " << decode_bin;
    return decode_bin == bin;
}

bool rsa_test4(const std::string &bin) {
    std::string codes = orz::rsa_mem_public_encode(mem_pub_key, bin);
    std::string decode_bin = orz::rsa_mem_private_decode(mem_key, codes);
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
    int N = 4096;
    int count = 0;
    for (int i = 0; i < N; ++i) {
        std::string bin = random_string();
        if (i % 2 == 0 && rsa_test1(bin)) ++count;
        if (i % 2 == 1 && rsa_test2(bin)) ++count;
        if (i % 2 == 2 && rsa_test3(bin)) ++count;
        if (i % 2 == 3 && rsa_test4(bin)) ++count;
    }

    ORZ_LOG(orz::INFO) << "Test count: " << N << ", Succeed count: " << count << ".";

    return 0;
}

