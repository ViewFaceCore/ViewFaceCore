//
// Created by kier on 2018/11/7.
//

#ifndef TENSORSTACK_MODULE_IO_FSTREAM_H
#define TENSORSTACK_MODULE_IO_FSTREAM_H

#include "stream.h"
#include <fstream>

namespace ts {
    class TS_DEBUG_API FileStreamReader : public StreamReader {
    public:
        using self = FileStreamReader;
        using supper = StreamReader;
        using std_stream = std::ifstream;

        FileStreamReader(const self &) = delete;

        self &operator=(const self &) = delete;

        FileStreamReader();

        explicit FileStreamReader(const std::string &path);

        void open(const std::string &path);

        bool is_open() const;

        void close();

        size_t read(void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
    };

    class TS_DEBUG_API FileStreamWriter : public StreamWriter {
    public:
        using self = FileStreamWriter;
        using supper = StreamWriter;
        using std_stream = std::ofstream;

        FileStreamWriter(const self &) = delete;

        self &operator=(const self &) = delete;

        FileStreamWriter();

        explicit FileStreamWriter(const std::string &path);

        void open(const std::string &path);

        bool is_open() const;

        void close();

        size_t write(const void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
    };

    class TS_DEBUG_API FileStream : public Stream {
    public:
        using self = FileStream;
        using supper = Stream;
        using std_stream = std::fstream;

        FileStream(const self &) = delete;

        self &operator=(const self &) = delete;

        FileStream();

        explicit FileStream(const std::string &path);

        void open(const std::string &path);

        bool is_open() const;

        void close();

        size_t read(void *buffer, size_t size) final;

        size_t write(const void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
    };
}


#endif //TENSORSTACK_MODULE_IO_FSTREAM_H
