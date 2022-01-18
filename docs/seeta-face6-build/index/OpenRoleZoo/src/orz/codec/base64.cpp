//
// Created by lby on 2018/1/10.
//

#include "orz/codec/base64.h"
#include "orz/utils/log.h"

namespace orz {
    static const char base64_encode_map[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static const int base64_decode_map[] = {
        //   0   1   2   3   4   5   6   7   8   9  A    B   C   D   E   F
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 1
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63, // 2
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, // 3
            -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, // 4
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, // 5
            -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, // 6
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1  // 7
    };

    std::string base64_encode(const std::string &bin) {
        std::string codes;
        std::string buff = bin;
        size_t un = buff.length() % 3;
        size_t extending = 0;
        if (un) extending = 3 - un;
        for (size_t i = 0; i < extending; ++i) buff.push_back('\0');
        for (size_t i = 2; i < buff.size(); i += 3) {
            int bin_code = (buff[i - 2] & 0xff) << 16;
            bin_code |= (buff[i - 1] & 0xff) << 8;
            bin_code |= (buff[i] & 0xff);
            codes.push_back(base64_encode_map[bin_code >> 18 & 0x3f]);
            codes.push_back(base64_encode_map[bin_code >> 12 & 0x3f]);
            codes.push_back(base64_encode_map[bin_code >> 6 & 0x3f]);
            codes.push_back(base64_encode_map[bin_code & 0x3f]);
        }
        for (size_t i = 0; i < extending; ++i) codes[codes.size() - 1 - i] = '=';
        return std::move(codes);
    }

    std::string base64_decode(const std::string &codes) {
        std::string bin;
        auto &buff = codes;
        int equal_sign_count = 0;
        int bin_code = 0;
        int bin_code_count = 0;
        for (size_t i = 0; i < buff.size(); ++i)
        {
            char ch = buff[i];
            if (ch & 0x80) ORZ_LOG(ERROR) << "unrecognized code: " << std::hex << ch << crash;
            auto idx = base64_decode_map[int(ch)];
            if (idx < 0)
            {
                if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
                else if (ch == '=') equal_sign_count++;
                else ORZ_LOG(ERROR) << "unrecognized code: " << std::hex << ch << crash;
            }
            if (equal_sign_count > 0 && idx >= 0) ORZ_LOG(ERROR) << "\"=\" appear in the middle of codes" << crash;
            bin_code = (bin_code << 6) | (idx & 0xff);
            if (++bin_code_count < 4) continue;
            bin.push_back(char(bin_code >> 16 & 0xff));
            bin.push_back(char(bin_code >> 8 & 0xff));
            bin.push_back(char(bin_code & 0xff));
            bin_code = 0;
            bin_code_count = 0;
        }
        if (bin_code != 0) ORZ_LOG(ERROR) << "length of codes is not a multiplier of 4" << crash;
        if (equal_sign_count > 2) ORZ_LOG(ERROR) << "equal sign count mismatch vs. " << equal_sign_count << crash;
        for (int i = 0; i < equal_sign_count; ++i) bin.pop_back();
        return std::move(bin);
    }
}
