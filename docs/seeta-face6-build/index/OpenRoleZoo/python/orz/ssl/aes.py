#!/usr/bin/env python
# coding: utf-8

from Crypto.Cipher import AES


class AESCrypto(object):
    def __init__(self, key, iv=None):
        # type: (str, str) -> None

        # only support AES
        self.__block_size = 16

        assert len(key) == self.__block_size

        self.key = key
        self.mode = AES.MODE_CBC

        if iv is None:
            iv = '\0' * self.__block_size

        assert len(iv) == self.__block_size

        self.iv = iv

    def encrypt(self, data, iv=None):
        # type: (bytes, bytes) -> bytes

        if iv is None:
            iv = self.iv

        assert len(iv) == self.__block_size

        crypto = AES.new(self.key, self.mode, iv)

        padded_data = self.__add_pkcs7_padding(data)

        return crypto.encrypt(padded_data)

    def decrypt(self, data, iv=None):
        # type: (bytes, bytes) -> bytes

        if iv is None:
            iv = self.iv

        assert len(iv) == self.__block_size

        crypto = AES.new(self.key, self.mode, iv)

        decrypted_data = crypto.decrypt(data)

        return self.__remove_pkcs7_padding(decrypted_data)

    @classmethod
    def __fake_tail(cls, data):
        # type: (bytes) -> bool

        if len(data) == 0:
            return False

        data_len = len(data)
        ch = data[-1]
        num = ord(ch)
        if num < data_len:
            for i in range(data_len - num, data_len):
                if data[i] != ch:
                    return False
            return True
        return False

    def __add_pkcs7_padding(self, data):
        # type: (bytes) -> bytes

        if len(data) == 0:
            return data

        tail_size = len(data) % self.__block_size
        if tail_size > 0:
            padding_size = self.__block_size - tail_size
            return data + padding_size * chr(padding_size).encode()
        elif self.__fake_tail(data):
            padding_size = self.__block_size
            return data + padding_size * chr(padding_size).encode()

        return data

    @classmethod
    def __remove_pkcs7_padding(cls, data):
        # type: (bytes) -> bytes

        if len(data) == 0:
            return data

        data_len = len(data)
        ch = data[-1]
        if isinstance(ch, int):
            num = ch
        else:
            num = ord(ch)
        if num < data_len:
            for i in range(data_len - num, data_len):
                if data[i] != ch:
                    return data
            if isinstance(ch, int):
                return data.rstrip(chr(ch).encode())
            else:
                return data.rstrip(ch)

        return data
