//
// Created by lby on 2018/1/16.
//

#include "orz/ssl/rsa.h"
#include "orz/utils/log.h"

#ifndef ORZ_WITH_OPENSSL

#include <functional>

#else   // ORZ_WITH_OPENSSL

#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <memory>
#include <cstring>
#include "orz/mem/need.h"

#endif  // !ORZ_WITH_OPENSSL

namespace orz {
    class rsa_key {
#ifndef ORZ_WITH_OPENSSL
#else   // !ORZ_WITH_OPENSSL
    public:
        using inner_type = rsa_st;
        enum key_type {
            PUBLIC,
            PRIVATE
        };

        rsa_key(inner_type *rsa, key_type type) : m_rsa(rsa), m_type(type) {}

        key_type type() const {
            return m_type;
        }

        inner_type *inner() {
            return m_rsa;
        }

        const inner_type *inner() const {
            return m_rsa;
        }

        inner_type *m_rsa = nullptr;
        key_type m_type;

        static rsa_key *load_public(const std::string &filename) {
#if _MSC_VER >= 1600
            FILE *key_file = nullptr;
            fopen_s(&key_file, filename.c_str(), "rb");
#else
            FILE *key_file = std::fopen(filename.c_str(), "rb");
#endif
            if (key_file == nullptr) return nullptr;
            need close_file(fclose, key_file);
            rsa_st *rsa = PEM_read_RSA_PUBKEY(key_file, nullptr, nullptr, nullptr);
            if (rsa == nullptr) return nullptr;
            return new rsa_key(rsa, PUBLIC);
        }

        static rsa_key *load_private(const std::string &filename) {
#if _MSC_VER >= 1600
            FILE *key_file = nullptr;
            fopen_s(&key_file, filename.c_str(), "rb");
#else
            FILE *key_file = std::fopen(filename.c_str(), "rb");
#endif
            if (key_file == nullptr) return nullptr;
            need close_file(fclose, key_file);
            rsa_st *rsa = PEM_read_RSAPrivateKey(key_file, nullptr, nullptr, nullptr);
            if (rsa == nullptr) return nullptr;
            return new rsa_key(rsa, PRIVATE);
        }

        static rsa_key *load_mem_public(const std::string &buffer) {
            auto key_str = buffer;
            auto key_length = key_str.size();
            for (size_t i = 64; i < key_length; i += 64) {
                if (key_str[i] != '\n') {
                    key_str.insert(i, "\n");
                }
                i++;
            }
            key_str.insert(0, "-----BEGIN PUBLIC KEY-----\n");
            key_str.append("\n-----END PUBLIC KEY-----\n");
            char *key_c_str = const_cast<char *>(key_str.c_str());
            BIO *bp = BIO_new_mem_buf(key_c_str, -1);
            if (bp == nullptr) {
                ORZ_LOG(ERROR) << "BIO_new_mem_buf failed.";
                return nullptr;
            }
            need free_bio(BIO_free_all, bp);

            rsa_st *rsa = PEM_read_bio_RSA_PUBKEY(bp, nullptr, nullptr, nullptr);
            if (rsa == nullptr) {
                ERR_load_crypto_strings();
                char err_str[512];
                ERR_error_string_n(ERR_get_error(), err_str, sizeof(err_str));
                ORZ_LOG(ERROR) << "load public key failed[" << err_str << "]";
                return nullptr;
            }
            return new rsa_key(rsa, PUBLIC);
        }

        static rsa_key *load_mem_private(const std::string &buffer) {
            auto key_str = buffer;
            auto key_length = key_str.size();
            for (size_t i = 64; i < key_length; i += 64) {
                if (key_str[i] != '\n') {
                    key_str.insert(i, "\n");
                }
                i++;
            }
            key_str.insert(0, "-----BEGIN RSA PRIVATE KEY-----\n");
            key_str.append("\n-----END RSA PRIVATE KEY-----\n");
            char *key_c_str = const_cast<char *>(key_str.c_str());
            BIO *bp = BIO_new_mem_buf(key_c_str, -1);
            if (bp == nullptr) {
                ORZ_LOG(ERROR) << "BIO_new_mem_buf failed.";
                return nullptr;
            }
            need free_bio(BIO_free_all, bp);

            rsa_st *rsa = PEM_read_bio_RSAPrivateKey(bp, nullptr, nullptr, nullptr);
            if (rsa == nullptr) {
                ERR_load_crypto_strings();
                char err_str[512];
                ERR_error_string_n(ERR_get_error(), err_str, sizeof(err_str));
                ORZ_LOG(ERROR) << "load private key failed[" << err_str << "]";
                return nullptr;
            }
            return new rsa_key(rsa, PRIVATE);
        }

        static void free(const rsa_key *key) {
            if (key) RSA_free(key->m_rsa);
        }

#endif  // !ORZ_WITH_OPENSSL
    };

    rsa_key *load_public_rsa_key(const std::string &filename) {
        return rsa_key::load_public(filename);
    }

    rsa_key *load_private_rsa_key(const std::string &filename) {
        return rsa_key::load_private(filename);
    }

    rsa_key *load_mem_public_rsa_key(const std::string &buffer) {
        return rsa_key::load_mem_public(buffer);
    }

    rsa_key *load_mem_private_rsa_key(const std::string &buffer) {
        return rsa_key::load_mem_private(buffer);
    }

    void free_rsa_key(const rsa_key *key) {
        rsa_key::free(key);
    }

    static std::string block_transfer(const std::string &data, size_t block_size,
                                      std::function<std::string(const std::string &)> transfer) {
        std::string rdata;
        std::string block;

        size_t index = 0;
        while (true) {
            if (index >= data.length()) break;
            block = data.substr(index, block_size);
            if (block.empty()) break;
            block = transfer(block);
            rdata.insert(rdata.end(), block.begin(), block.end());
            index += block_size;
        }
        return std::move(rdata);
    }

    std::string rsa_private_encode(const std::string &filename, const std::string &data) {
        auto key = load_private_rsa_key(filename);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_private_encode(key, data);
    }

    std::string rsa_public_decode(const std::string &filename, const std::string &data) {
        auto key = load_public_rsa_key(filename);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_public_decode(key, data);
    }

    std::string rsa_public_encode(const std::string &filename, const std::string &data) {
        auto key = load_public_rsa_key(filename);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_public_encode(key, data);
    }

    std::string rsa_private_decode(const std::string &filename, const std::string &data) {
        auto key = load_private_rsa_key(filename);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_private_decode(key, data);
    }

    std::string rsa_mem_private_encode(const std::string &buffer, const std::string &data) {
        auto key = load_mem_private_rsa_key(buffer);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_private_encode(key, data);
    }

    std::string rsa_mem_public_decode(const std::string &buffer, const std::string &data) {
        auto key = load_mem_public_rsa_key(buffer);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_public_decode(key, data);
    }

    std::string rsa_mem_public_encode(const std::string &buffer, const std::string &data) {
        auto key = load_mem_public_rsa_key(buffer);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_public_encode(key, data);
    }

    std::string rsa_mem_private_decode(const std::string &buffer, const std::string &data) {
        auto key = load_mem_private_rsa_key(buffer);
        if (key == nullptr) return std::string();
        need free_key(free_rsa_key, key);

        return rsa_private_decode(key, data);
    }


    std::string rsa_private_encode(rsa_key *key, const std::string &data) {
        if (key == nullptr) return std::string();

        auto rsa_len = RSA_size(key->inner());
        auto block_size = rsa_len - 11; // RSA_PKCS1_PADDING

        try { return block_transfer(data, size_t(block_size), std::bind(rsa_encode_block, key, std::placeholders::_1)); }
        catch (const Exception &) { return std::string(); }
    }

    std::string rsa_public_decode(rsa_key *key, const std::string &data) {
        if (key == nullptr) return std::string();

        auto rsa_len = RSA_size(key->inner());
        auto block_size = rsa_len;

        try { return block_transfer(data, size_t(block_size), std::bind(rsa_decode_block, key, std::placeholders::_1)); }
        catch (const Exception &) { return std::string(); }
    }

    std::string rsa_public_encode(rsa_key *key, const std::string &data) {
        if (key == nullptr) return std::string();

        auto rsa_len = RSA_size(key->inner());
        auto block_size = rsa_len - 11; // RSA_PKCS1_PADDING

        try { return block_transfer(data, size_t(block_size), std::bind(rsa_encode_block, key, std::placeholders::_1)); }
        catch (const Exception &) { return std::string(); }
    }

    std::string rsa_private_decode(rsa_key *key, const std::string &data) {
        if (key == nullptr) return std::string();

        auto rsa_len = RSA_size(key->inner());
        auto block_size = rsa_len;

        try { return block_transfer(data, size_t(block_size), std::bind(rsa_decode_block, key, std::placeholders::_1)); }
        catch (const Exception &) { return std::string(); }
    }

    std::string rsa_encode_block(rsa_key *key, const std::string &data) {
#ifndef ORZ_WITH_OPENSSL

#else   // ORZ_WITH_OPENSSL
        RSA *rsa = key->inner();
        auto rsa_len = RSA_size(rsa);
        std::unique_ptr<char[]> rdata(new char[rsa_len + 1]);
        std::memset(rdata.get(), 0, rsa_len + 1);
        int rsize = 0;
        switch (key->type()) {
            case rsa_key::PUBLIC:
                rsize = RSA_public_encrypt(
                        static_cast<int>(data.size()),
                        reinterpret_cast<const unsigned char *>(data.data()),
                        reinterpret_cast<unsigned char *>(rdata.get()),
                        rsa, RSA_PKCS1_PADDING /* RSA_PKCS1_PADDING RSA_NO_PADDING */);
                break;
            case rsa_key::PRIVATE:
                rsize = RSA_private_encrypt(
                        static_cast<int>(data.size()),
                        reinterpret_cast<const unsigned char *>(data.data()),
                        reinterpret_cast<unsigned char *>(rdata.get()),
                        rsa, RSA_PKCS1_PADDING /* RSA_PKCS1_PADDING RSA_NO_PADDING */);
                break;
        }
        if (rsize < 0) {
            ERR_print_errors_fp(stdout);
            ORZ_LOG(ERROR) << "OpenSSL RAS error." << crash;
        }
        return std::string(rdata.get(), static_cast<size_t >(rsize));
#endif
    }

    std::string rsa_decode_block(rsa_key *key, const std::string &data) {
#ifndef ORZ_WITH_OPENSSL

#else   // ORZ_WITH_OPENSSL
        RSA *rsa = key->inner();
        auto rsa_len = RSA_size(rsa);
        std::unique_ptr<char[]> rdata(new char[rsa_len + 1]);
        std::memset(rdata.get(), 0, rsa_len + 1);
        int rsize = 0;
        switch (key->type()) {
            case rsa_key::PUBLIC:
                rsize = RSA_public_decrypt(
                        static_cast<int>(data.size()),
                        reinterpret_cast<const unsigned char *>(data.data()),
                        reinterpret_cast<unsigned char *>(rdata.get()),
                        rsa, RSA_PKCS1_PADDING /* RSA_PKCS1_PADDING RSA_NO_PADDING */);
                break;
            case rsa_key::PRIVATE:
                rsize = RSA_private_decrypt(
                        static_cast<int>(data.size()),
                        reinterpret_cast<const unsigned char *>(data.data()),
                        reinterpret_cast<unsigned char *>(rdata.get()),
                        rsa, RSA_PKCS1_PADDING /* RSA_PKCS1_PADDING RSA_NO_PADDING */);
                break;
        }
        if (rsize < 0) {
            ERR_print_errors_fp(stdout);
            ORZ_LOG(ERROR) << "OpenSSL RAS error." << crash;
        }
        return std::string(rdata.get(), static_cast<size_t >(rsize));
#endif
    }
}
