//
// Created by kier on 2018/11/7.
//

#include <module/io/sstream.h>

#include "module/io/sstream.h"

namespace ts {

    StringStreamReader::string StringStreamReader::str() const {
        return m_stream.str();
    }

    void StringStreamReader::clear() {
        m_stream.str("");
    }

    size_t StringStreamReader::read(void *buffer, size_t size) {
        m_stream.read(reinterpret_cast<char *>(buffer), size);
        return size_t(m_stream.gcount());
    }

    StringStreamReader::StringStreamReader(const StringStreamReader::string &buffer)
            : m_stream(buffer) {}

    void StringStreamReader::str(const StringStreamReader::string &buffer) {
        m_stream.str(buffer);
    }

    StringStreamWriter::string StringStreamWriter::str() const {
        return m_stream.str();
    }

    void StringStreamWriter::clear() {
        m_stream.str("");
    }

    size_t StringStreamWriter::write(const void *buffer, size_t size) {
        m_stream.write(reinterpret_cast<const char *>(buffer), size);
        return m_stream.bad() ? 0 : size;
    }

    StringStreamWriter::StringStreamWriter(const StringStreamWriter::string &buffer)
            : m_stream(buffer) {}

    void StringStreamWriter::str(const StringStreamWriter::string &buffer) {
        m_stream.str();
    }

    StringStream::string StringStream::str() const {
        return m_stream.str();
    }

    void StringStream::clear() {
        m_stream.str("");
    }

    size_t StringStream::read(void *buffer, size_t size) {
        m_stream.read(reinterpret_cast<char *>(buffer), size);
        return size_t(m_stream.gcount());
    }

    size_t StringStream::write(const void *buffer, size_t size) {
        m_stream.write(reinterpret_cast<const char *>(buffer), size);
        return m_stream.bad() ? 0 : size;
    }

    StringStream::StringStream(const StringStream::string &buffer)
            : m_stream(buffer) {}

    void StringStream::str(const StringStream::string &buffer) {
        m_stream.str(buffer);
    }
}
