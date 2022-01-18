//
// Created by kier on 2018/12/3.
//

#include <module/header.h>

#include "module/header.h"

namespace ts {
    size_t Header::serialize(StreamWriter &stream) const {
        size_t writen_size = 0;
        writen_size += binio::write<uint32_t>(stream, fake);
        writen_size += binio::write<uint32_t>(stream, code);
        writen_size += binio::write<char>(stream, data.data(), data.size());
        return writen_size;
    }

    size_t Header::externalize(StreamReader &stream) {
        size_t read_size = 0;
        read_size += binio::read<uint32_t>(stream, fake);
        read_size += binio::read<uint32_t>(stream, code);
        read_size += binio::read<char>(stream, data.data(), data.size());
        return read_size;
    }
}