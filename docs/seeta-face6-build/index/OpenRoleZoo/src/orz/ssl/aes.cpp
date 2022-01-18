//
// Created by lby on 2018/1/15.
//

#include "orz/ssl/aes.h"

#include "orz/utils/log.h"

#ifndef ORZ_WITH_OPENSSL

#include <cstring>
#include <cstdint>
#include <cassert>
#include <memory>

#else   // ORZ_WITH_OPENSSL

#include <openssl/aes.h>
#include <memory>
#include <cstring>

#endif  // !ORZ_WITH_OPENSSL

namespace orz {

#ifndef ORZ_WITH_OPENSSL

    class block4x4 {
    public:
        block4x4(char *data) : m_data(data) {}

        // block4x4(const char *data) : m_data(const_cast<char *>(data)) {}

        char &operator()(int row, int col) {
            return m_data[row * 4 + col];
        }

        unsigned char &u(int row, int col) {
            return *(reinterpret_cast<unsigned char *>(&operator()(row, col)));
        }

        uint32_t &row(int n) {
            return *(reinterpret_cast<uint32_t *>(&operator()(n, 0)));
        }

        char *data() { return m_data; }

    private:
        char *m_data;
    };

    static void left_shift_loop(uint32_t &lhs, int rhs) {
        lhs = (lhs << rhs) | (lhs >> (32 - rhs));
    }

    static void right_shift_loop(uint32_t &lhs, int rhs) {
        lhs = (lhs >> rhs) | (lhs << (32 - rhs));
    }

    static void shift_rows_encode(block4x4 block) {
        left_shift_loop(block.row(1), 8);
        left_shift_loop(block.row(2), 16);
        left_shift_loop(block.row(3), 24);
    }

    static void shift_rows_decode(block4x4 block) {
        right_shift_loop(block.row(1), 8);
        right_shift_loop(block.row(2), 16);
        right_shift_loop(block.row(3), 24);
    }

    static unsigned char XTIME(unsigned char x) {
        return static_cast<unsigned char>(((x << 1) ^ ((x & 0x80) ? 0x1b : 0x00)) & 0xff);
    }

    static unsigned char multiply(unsigned char a, unsigned char b) {
        unsigned char temp[8] = {a};
        unsigned char tempmultiply = 0x00;
        int i = 0;
        for (i = 1; i < 8; i++) {
            temp[i] = XTIME(temp[i - 1]);
        }
        tempmultiply = static_cast<unsigned char>(((b & 0x01) * a) & 0xff);
        for (i = 1; i <= 7; i++) {
            tempmultiply ^= (((b >> i) & 0x01) * temp[i]);
        }
        return tempmultiply;
    }

    static char multiply(char a, char b) {
        return static_cast<char>(
                multiply(static_cast<unsigned char>(a),
                         static_cast<unsigned char>(b)));
    }

    // a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4
    static char vector4_mul(
            char a1, char a2, char a3, char a4,
            char b1, char b2, char b3, char b4
    ) {
        return multiply(a1, b1) ^ multiply(a2, b2) ^ multiply(a3, b3) ^ multiply(a4, b4);
    }

    // a * b => c
    static void metrix_mul(block4x4 a, block4x4 b, block4x4 c) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                c(row, col) = vector4_mul(
                        a(row, 0),a(row, 1),a(row, 2),a(row, 3),
                        b(0, col),b(1, col),b(2, col),b(3, col)
                );
            }
        }
    }

    static char encode_matrix[] = {
            0x02, 0x03, 0x01, 0x01,
            0x01, 0x02, 0x03, 0x01,
            0x01, 0x01, 0x02, 0x03,
            0x03, 0x01, 0x01, 0x02,
    };

    static char decode_matrix[] = {
            0x0E, 0x0B, 0x0D, 0x09,
            0x09, 0x0E, 0x0B, 0x0D,
            0x0D, 0x09, 0x0E, 0x0B,
            0x0B, 0x0D, 0x09, 0x0E,
    };

    static void mix_columns_encode(block4x4 block) {
        block4x4 encode_block = encode_matrix;
        char result[16];
        block4x4 result_blobk = result;
        metrix_mul(encode_block, block, result_blobk);
        std::memcpy(block.data(), result, 16);
    }

    static void mix_columns_decode(block4x4 block) {
        block4x4 decode_block = decode_matrix;
        char result[16];
        block4x4 result_blobk = result;
        metrix_mul(decode_block, block, result_blobk);
        std::memcpy(block.data(), result, 16);
    }

#endif  // !ORZ_WITH_OPENSSL

    std::string
    aes128_encode_block(const std::string &key, CRYPTO_MODE mode, const std::string &data, const std::string &iv) {
#ifndef ORZ_WITH_OPENSSL
#warning Only support OpenSSL, please recomiple with -DORZ_WITH_OPENSSL
        std::unique_ptr<char[]> rdata(new char[data.size()]);
        std::memcpy(rdata.get(), data.data(), data.size());
        shift_rows_encode(rdata.get());
        mix_columns_encode(rdata.get());
        return std::string(rdata.get(), data.size());
#else   // ORZ_WITH_OPENSSL
        if (key.length() != 16) ORZ_LOG(ERROR) << "key.length should be 16 vs. " << key.length() << crash;
        if (data.length() % AES_BLOCK_SIZE != 0)
            ORZ_LOG(ERROR) << "length of data is not a multiplier of " << AES_BLOCK_SIZE << crash;
        std::string iv_copy = iv;
        unsigned char iv_buff[AES_BLOCK_SIZE];
        if (mode == CBC) {
            if (iv_copy.empty()) iv_copy = std::string(AES_BLOCK_SIZE, 0);
            if (iv_copy.length() != AES_BLOCK_SIZE)
                ORZ_LOG(ERROR) << "iv.length should be " << AES_BLOCK_SIZE << " vs. " << iv_copy.length() << crash;
            std::memcpy(iv_buff, iv_copy.data(), AES_BLOCK_SIZE);
        }
        AES_KEY aes = {0};
        if (AES_set_encrypt_key(reinterpret_cast<const unsigned char *>(key.data()), 128, &aes)) {
            ORZ_LOG(ERROR) << "openssl: can not init key: " << key << crash;
        }
        std::unique_ptr<char[]> rdata(new char[data.size()]);
        switch (mode) {
            case CBC:
                AES_cbc_encrypt(
                        reinterpret_cast<const unsigned char *>(data.data()),
                        reinterpret_cast<unsigned char *>(rdata.get()),
                        data.size(), &aes, iv_buff, AES_ENCRYPT
                );
        }
        return std::string(rdata.get(), data.size());
#endif  // !ORZ_WITH_OPENSSL
    }

    std::string
    aes128_decode_block(const std::string &key, CRYPTO_MODE mode, const std::string &data, const std::string &iv) {
#ifndef ORZ_WITH_OPENSSL
#warning Only support OpenSSL, please recomiple with -ORZ_WITH_OPENSSL
        std::unique_ptr<char[]> rdata(new char[data.size()]);
        std::memcpy(rdata.get(), data.data(), data.size());
        mix_columns_decode(rdata.get());
        shift_rows_decode(rdata.get());
        return std::string(rdata.get(), data.size());
#else   // ORZ_WITH_OPENSSL
        if (key.length() != 16) ORZ_LOG(ERROR) << "key.length should be 16 vs. " << key.length() << crash;
        if (data.length() % AES_BLOCK_SIZE != 0)
            ORZ_LOG(ERROR) << "length of data is not a multiplier of " << AES_BLOCK_SIZE << crash;
        std::string iv_copy = iv;
        unsigned char iv_buff[AES_BLOCK_SIZE];
        if (mode == CBC) {
            if (iv_copy.empty()) iv_copy = std::string(AES_BLOCK_SIZE, 0);
            if (iv_copy.length() != AES_BLOCK_SIZE)
                ORZ_LOG(ERROR) << "iv.length should be " << AES_BLOCK_SIZE << " vs. " << iv_copy.length() << crash;
            std::memcpy(iv_buff, iv_copy.data(), AES_BLOCK_SIZE);
        }
        AES_KEY aes = {0};
        if (AES_set_decrypt_key(reinterpret_cast<const unsigned char *>(key.data()), 128, &aes)) {
            ORZ_LOG(ERROR) << "openssl: can not init key: " << key << crash;
        }
        std::unique_ptr<char[]> rdata(new char[data.size()]);
        switch (mode) {
            case CBC:
                AES_cbc_encrypt(
                        reinterpret_cast<const unsigned char *>(data.data()),
                        reinterpret_cast<unsigned char *>(rdata.get()),
                        data.size(), &aes, iv_buff, AES_DECRYPT
                );
        }
        return std::string(rdata.get(), data.size());
#endif  // !ORZ_WITH_OPENSSL
    }

    static bool feak_tail(const std::string &data) {
        size_t len = data.size();
        char ch = data.back();
        size_t num = static_cast<size_t>(ch);
        if (num < len) {
            for (auto i = len - num; i < len; ++i) {
                if (data[i] != ch) return false;
            }
            return true;
        }
        return false;
    }

    void aes128_PKCS7_add_padding(std::string &data) {
        static size_t block_size = 16;
        auto tail_size = data.size() % block_size;
        if (tail_size > 0) {
            auto padding_size = block_size - tail_size;
            data.insert(data.end(), padding_size, (unsigned char) (padding_size));
        } else if (feak_tail(data)) {
            data.insert(data.end(), block_size, (unsigned char) (block_size));
        }
    }

    void aes128_PKCS7_reamove_padding(std::string &data) {
        size_t len = data.size();
        char ch = data.back();
        size_t num = static_cast<size_t>(ch);
        if (num < len) {
            for (auto i = len - num; i < len; ++i) {
                if (data[i] != ch) return;
            }
            data.erase(len - num);
        }
    }

    std::string aes128_encode(const std::string &key, CRYPTO_MODE mode, const std::string &data,
                              const std::string &iv) {
        auto padded_data = data;
        aes128_PKCS7_add_padding(padded_data);
        return aes128_encode_block(key, mode, padded_data, iv);
    }

    std::string aes128_decode(const std::string &key, CRYPTO_MODE mode, const std::string &data,
                              const std::string &iv) {
        auto padded_data = aes128_decode_block(key, mode, data, iv);
        aes128_PKCS7_reamove_padding(padded_data);
        return std::move(padded_data);
    }

    binary
    aes128_encode_block(const std::string &key, CRYPTO_MODE mode, const binary &data, const std::string &iv) {
#ifndef ORZ_WITH_OPENSSL
        #warning Only support OpenSSL, please recomiple with -DORZ_WITH_OPENSSL
        std::unique_ptr<char[]> rdata(new char[data.size()]);
        std::memcpy(rdata.get(), data.data(), data.size());
        shift_rows_encode(rdata.get());
        mix_columns_encode(rdata.get());
        return std::string(rdata.get(), data.size());
#else   // ORZ_WITH_OPENSSL
        if (key.length() != 16) ORZ_LOG(ERROR) << "key.length should be 16 vs. " << key.length() << crash;
        if (data.size() % AES_BLOCK_SIZE != 0)
            ORZ_LOG(ERROR) << "length of data is not a multiplier of " << AES_BLOCK_SIZE << crash;
        std::string iv_copy = iv;
        unsigned char iv_buff[AES_BLOCK_SIZE];
        if (mode == CBC) {
            if (iv_copy.empty()) iv_copy = std::string(AES_BLOCK_SIZE, 0);
            if (iv_copy.length() != AES_BLOCK_SIZE)
                ORZ_LOG(ERROR) << "iv.length should be " << AES_BLOCK_SIZE << " vs. " << iv_copy.length() << crash;
            std::memcpy(iv_buff, iv_copy.data(), AES_BLOCK_SIZE);
        }
        AES_KEY aes = {0};
        if (AES_set_encrypt_key(reinterpret_cast<const unsigned char *>(key.data()), 128, &aes)) {
            ORZ_LOG(ERROR) << "openssl: can not init key: " << key << crash;
        }
        binary rdata(data.size());
        switch (mode) {
            case CBC:
                AES_cbc_encrypt(
                        data.data<unsigned char>(),
                        rdata.data<unsigned char>(),
                        data.size(), &aes, iv_buff, AES_ENCRYPT
                );
        }
        return std::move(rdata);
#endif  // !ORZ_WITH_OPENSSL
    }

    binary
    aes128_decode_block(const std::string &key, CRYPTO_MODE mode, const binary &data, const std::string &iv) {
#ifndef ORZ_WITH_OPENSSL
        #warning Only support OpenSSL, please recomiple with -ORZ_WITH_OPENSSL
        std::unique_ptr<char[]> rdata(new char[data.size()]);
        std::memcpy(rdata.get(), data.data(), data.size());
        mix_columns_decode(rdata.get());
        shift_rows_decode(rdata.get());
        return std::string(rdata.get(), data.size());
#else   // ORZ_WITH_OPENSSL
        if (key.length() != 16) ORZ_LOG(ERROR) << "key.length should be 16 vs. " << key.length() << crash;
        if (data.size() % AES_BLOCK_SIZE != 0)
            ORZ_LOG(ERROR) << "length of data is not a multiplier of " << AES_BLOCK_SIZE << crash;
        std::string iv_copy = iv;
        unsigned char iv_buff[AES_BLOCK_SIZE];
        if (mode == CBC) {
            if (iv_copy.empty()) iv_copy = std::string(AES_BLOCK_SIZE, 0);
            if (iv_copy.length() != AES_BLOCK_SIZE)
                ORZ_LOG(ERROR) << "iv.length should be " << AES_BLOCK_SIZE << " vs. " << iv_copy.length() << crash;
            std::memcpy(iv_buff, iv_copy.data(), AES_BLOCK_SIZE);
        }
        AES_KEY aes = {0};
        if (AES_set_decrypt_key(reinterpret_cast<const unsigned char *>(key.data()), 128, &aes)) {
            ORZ_LOG(ERROR) << "openssl: can not init key: " << key << crash;
        }
        binary rdata(data.size());
        switch (mode) {
            case CBC:
                AES_cbc_encrypt(
                        data.data<unsigned char>(),
                        rdata.data<unsigned char>(),
                        data.size(), &aes, iv_buff, AES_DECRYPT
                );
        }
        return std::move(rdata);
#endif  // !ORZ_WITH_OPENSSL
    }

    binary aes128_encode(const std::string &key, CRYPTO_MODE mode, const binary &data, const std::string &iv) {
        if (data.size() == 0) return data;

        auto padded_data = data;
        aes128_PKCS7_add_padding(padded_data);
        return aes128_encode_block(key, mode, padded_data, iv);
    }

    binary aes128_decode(const std::string &key, CRYPTO_MODE mode, const binary &data, const std::string &iv) {
        if (data.size() == 0) return data;

        auto padded_data = aes128_decode_block(key, mode, data, iv);
        aes128_PKCS7_reamove_padding(padded_data);
        return std::move(padded_data);
    }

    static bool feak_tail(const binary &data) {
        if (data.size() == 0) return false;

        size_t len = data.size();
        char ch = data.data<char>()[data.size() - 1];
        size_t num = static_cast<size_t>(ch);
        if (num < len) {
            for (auto i = len - num; i < len; ++i) {
                if (data.data<char>()[i] != ch) return false;
            }
            return true;
        }
        return false;
    }

    void aes128_PKCS7_add_padding(binary &data) {
        if (data.size() == 0) return;

        static size_t block_size = 16;
        auto tail_size = data.size() % block_size;
        size_t padding_size = 0;
        if (tail_size > 0) {
            padding_size = block_size - tail_size;
        } else if (feak_tail(data)) {
            padding_size = block_size;
        }

        if (padding_size == 0) return;

        std::string padding( padding_size, (unsigned char) (padding_size));
        data.set_pos(binary::pos::end, 0);
        data.write(padding.data(), padding.size());
    }

    void aes128_PKCS7_reamove_padding(binary &data) {
        if (data.size() == 0) return;

        size_t len = data.size();
        char ch = data.data<char>()[data.size() - 1];
        size_t num = static_cast<size_t>(ch);
        if (num < len) {
            for (auto i = len - num; i < len; ++i) {
                if (data.data<char>()[i] != ch) return;
            }
            data.resize(len - num);
        }
    }
}

