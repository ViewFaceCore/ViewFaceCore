//
// Created by lby on 2018/1/16.
//

#ifndef ORZ_RSA_H
#define ORZ_RSA_H

#include <string>

namespace orz {
    class rsa_key;

    rsa_key *load_public_rsa_key(const std::string &filename);

    rsa_key *load_private_rsa_key(const std::string &filename);

    rsa_key *load_mem_public_rsa_key(const std::string &buffer);

    rsa_key *load_mem_private_rsa_key(const std::string &buffer);

    void free_rsa_key(const rsa_key *key);

    // encode with OPENSSL_PKCS1_PADDING, data length can not greater than 117
    std::string rsa_encode_block(rsa_key *key, const std::string &data);

    // decode with OPENSSL_PKCS1_PADDING, data length should be 128
    std::string rsa_decode_block(rsa_key *key, const std::string &data);

    std::string rsa_private_encode(rsa_key *key, const std::string &data);

    std::string rsa_public_decode(rsa_key *key, const std::string &data);

    std::string rsa_public_encode(rsa_key *keye, const std::string &data);

    std::string rsa_private_decode(rsa_key *key, const std::string &data);

    std::string rsa_private_encode(const std::string &filename, const std::string &data);

    std::string rsa_public_decode(const std::string &filename, const std::string &data);

    std::string rsa_public_encode(const std::string &filename, const std::string &data);

    std::string rsa_private_decode(const std::string &filename, const std::string &data);

    std::string rsa_mem_private_encode(const std::string &buffer, const std::string &data);

    std::string rsa_mem_public_decode(const std::string &buffer, const std::string &data);

    std::string rsa_mem_public_encode(const std::string &buffer, const std::string &data);

    std::string rsa_mem_private_decode(const std::string &buffer, const std::string &data);
}

#endif //ORZ_RSA_H
