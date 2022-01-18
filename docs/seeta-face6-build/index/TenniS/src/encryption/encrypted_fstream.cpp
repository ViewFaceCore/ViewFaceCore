//
// Created by kier on 2019/2/27.
//

#include <encryption/encrypted_fstream.h>

#include "encryption/encrypted_fstream.h"

#include "aes_fstream.h"

namespace ts {

    class EncryptedFileStreamReader::Implement {
    public:
        Implement(const std::string &path, const std::string &key) : stream(path, key) {}

        AESFileStreamReader stream;
    };

    EncryptedFileStreamReader::EncryptedFileStreamReader(const std::string &path, const std::string &key)
        : m_impl(path, key) {}

    bool EncryptedFileStreamReader::is_open() const {
        return m_impl->stream.is_open();
    }

    void EncryptedFileStreamReader::close() {
        m_impl->stream.close();
    }

    size_t EncryptedFileStreamReader::read(void *buffer, size_t size) {
        return m_impl->stream.read(buffer, size);
    }

    EncryptedFileStreamReader::~EncryptedFileStreamReader() {
        // do nothing
    }

    class EncryptedFileStreamWriter::Implement {
    public:
        Implement(const std::string &path, const std::string &key) : stream(path, key) {}

        AESFileStreamWriter stream;
    };

    EncryptedFileStreamWriter::EncryptedFileStreamWriter(const std::string &path, const std::string &key)
            : m_impl(path, key) {}

    bool EncryptedFileStreamWriter::is_open() const {
        return m_impl->stream.is_open();
    }

    void EncryptedFileStreamWriter::close() {
        m_impl->stream.close();
    }

    EncryptedFileStreamWriter::~EncryptedFileStreamWriter() {
        // do nothing
    }

    size_t EncryptedFileStreamWriter::write(const void *buffer, size_t size) {
        return m_impl->stream.write(buffer, size);
    }
}
