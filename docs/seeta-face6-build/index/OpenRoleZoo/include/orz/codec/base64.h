//
// Created by lby on 2018/1/10.
//

#ifndef ORZ_CODEC_BASE64_H
#define ORZ_CODEC_BASE64_H

#include <string>

namespace orz {
    std::string base64_encode(const std::string &bin);
    std::string base64_decode(const std::string &codes);
}

#endif //ORZ_CODEC_BASE64_H
