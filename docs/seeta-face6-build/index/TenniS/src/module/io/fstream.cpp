//
// Created by kier on 2018/11/7.
//

#include <module/io/fstream.h>

#include "module/io/fstream.h"

namespace ts {
    bool FileStreamReader::is_open() const {
        return m_stream.is_open();
    }

    size_t FileStreamReader::read(void *buffer, size_t size) {
        m_stream.read(reinterpret_cast<char *>(buffer), size);
        return size_t(m_stream.gcount());
    }

    FileStreamReader::FileStreamReader(const std::string &path)
            : m_stream(path, std::ios::binary) {}

    void FileStreamReader::open(const std::string &path) {
        m_stream.open(path, std::ios::binary);
    }

    void FileStreamReader::close() {
        m_stream.close();
    }

    FileStreamReader::FileStreamReader() = default;

    bool FileStreamWriter::is_open() const {
        return m_stream.is_open();
    }

    size_t FileStreamWriter::write(const void *buffer, size_t size) {
        m_stream.write(reinterpret_cast<const char *>(buffer), size);
        return m_stream.bad() ? 0 : size;
    }

    FileStreamWriter::FileStreamWriter(const std::string &path)
            : m_stream(path, std::ios::binary) {}

    void FileStreamWriter::open(const std::string &path) {
        return m_stream.open(path, std::ios::binary);
    }

    void FileStreamWriter::close() {
        m_stream.close();
    }

    FileStreamWriter::FileStreamWriter() = default;

    bool FileStream::is_open() const {
        return m_stream.is_open();
    }

    size_t FileStream::read(void *buffer, size_t size) {
        m_stream.read(reinterpret_cast<char *>(buffer), size);
        return size_t(m_stream.gcount());
    }

    size_t FileStream::write(const void *buffer, size_t size) {
        m_stream.write(reinterpret_cast<const char *>(buffer), size);
        return m_stream.bad() ? 0 : size;
    }

    FileStream::FileStream(const std::string &path)
            : m_stream(path, std::ios::binary) {}

    void FileStream::open(const std::string &path) {
        m_stream.open(path, std::ios::binary);
    }

    void FileStream::close() {
        m_stream.close();
    }

    FileStream::FileStream() = default;
}