//
// Created by kier on 2018/11/7.
//

#ifndef TENSORSTACK_MODULE_IO_STREAM_H
#define TENSORSTACK_MODULE_IO_STREAM_H


#include <cstddef>
#include <string>

#include <utils/api.h>


namespace ts {
    class TS_DEBUG_API StreamReader {
    public:
        using self = StreamReader;

        virtual size_t read(void *buffer, size_t size) = 0;
    };

    class TS_DEBUG_API StreamWriter {
    public:
        using self = StreamWriter;

        virtual size_t write(const void *buffer, size_t size) = 0;
    };

    class TS_DEBUG_API Stream : public StreamWriter, public StreamReader {
    public:
        using self = Stream;
    };

    namespace binio {
        template<typename T>
        size_t read(StreamReader &stream, T &buffer) {
            return stream.read(&buffer, sizeof(T));
        }

        template<typename T>
        size_t read(StreamReader &stream, T *buffer, size_t size) {
            return stream.read(buffer, sizeof(T) * size);
        }

        template<typename T, size_t _Size>
        size_t read(StreamReader &stream, T (&buffer)[_Size]) {
            return stream.read(buffer, sizeof(T) * _Size);
        }

        template<typename T>
        size_t write(StreamWriter &stream, const T &buffer) {
            return stream.write(&buffer, sizeof(T));
        }

        template<typename T>
        size_t write(StreamWriter &stream, const T *buffer, size_t size) {
            return stream.write(buffer, sizeof(T) * size);
        }

        template<typename T, size_t _Size>
        size_t write(StreamWriter &stream, const T (&buffer)[_Size]) {
            return stream.write(buffer, sizeof(T) * _Size);
        }
    }
}


#endif //TENSORSTACK_MODULE_IO_STREAM_H
