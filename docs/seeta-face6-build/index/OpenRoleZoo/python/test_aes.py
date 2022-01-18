#!/usr/bin/env python
# coding: UTF-8

from orz.ssl.aes import AESCrypto

import random


def random_string():
    size = int(random.random() % 1024)
    bin = ''
    for i in xrange(size):
        bin += chr(int(random.random() % 256))
    return bin


def aes_test(bin):
    crypto = AESCrypto('seetatech0123456')
    codes = crypto.encrypt(bin)
    decode_bin = crypto.decrypt(codes)

    return decode_bin == bin


if __name__ == '__main__':
    random.seed(7726)

    N = 32768
    count = 0
    for i in xrange(N):
        bin = random_string()
        if aes_test(bin):
            count += 1

    print("Test count: %d, Succeed count: %d." % (N, count))
