//
// Created by kier on 2018/11/7.
//

#ifndef TENSORSTACK_MODULE_IO_SSTREAM_H
#define TENSORSTACK_MODULE_IO_SSTREAM_H


#include "stream.h"
#include <sstream>

namespace ts {
    class TS_DEBUG_API StringStreamReader : public StreamReader {
    public:
        using self = StringStreamReader;
        using supper = StreamReader;
        using std_stream = std::istringstream;
        using string = std::string;

        StringStreamReader(const self &) = delete;

        self &operator=(const self &) = delete;

        StringStreamReader() = default;

        explicit StringStreamReader(const string &buffer);

        void str(const string &buffer);

        string str() const;

        void clear();

        size_t read(void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
    };

    class TS_DEBUG_API StringStreamWriter : public StreamWriter {
    public:
        using self = StringStreamWriter;
        using supper = StreamWriter;
        using std_stream = std::ostringstream;
        using string = std::string;

        StringStreamWriter(const self &) = delete;

        self &operator=(const self &) = delete;

        StringStreamWriter() = default;

        explicit StringStreamWriter(const string &buffer);

        void str(const string &buffer);

        string str() const;

        void clear();

        size_t write(const void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
    };

    class TS_DEBUG_API StringStream : public Stream {
    public:
        using self = StringStream;
        using supper = Stream;
        using std_stream = std::stringstream;
        using string = std::string;

        StringStream(const self &) = delete;

        self &operator=(const self &) = delete;

        StringStream() = default;

        explicit StringStream(const string &buffer);

        void str(const string &buffer);

        string str() const;

        void clear();

        size_t read(void *buffer, size_t size) final;

        size_t write(const void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
    };
}


#endif //TENSORSTACK_MODULE_IO_SSTREAM_H
