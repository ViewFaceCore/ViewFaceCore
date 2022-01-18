//
// Created by kier on 2019/2/27.
//

#ifndef TENSORSTACK_ENCRYPTION_ENCRYPTED_FSTREAM_H
#define TENSORSTACK_ENCRYPTION_ENCRYPTED_FSTREAM_H


#include "module/io/stream.h"
#include "utils/implement.h"

namespace ts {
    class TS_DEBUG_API EncryptedFileStreamReader : public StreamReader {
    public:
        using self = EncryptedFileStreamReader;
        using supper = StreamReader;
        using std_stream = std::ifstream;

        EncryptedFileStreamReader(const self &) = delete;

        self &operator=(const self &) = delete;

        EncryptedFileStreamReader() = delete;

        /**
         *
         * @param path file path
         * @param key length over 32 will be ignore
         */
        explicit EncryptedFileStreamReader(const std::string &path, const std::string &key);

        ~EncryptedFileStreamReader();

        bool is_open() const;

        void close();

        size_t read(void *buffer, size_t size) final;

    private:
        class Implement;
        Declare<Implement> m_impl;
    };


    class TS_DEBUG_API EncryptedFileStreamWriter : public StreamWriter {
    public:
        using self = EncryptedFileStreamWriter;
        using supper = StreamWriter;
        using std_stream = std::ofstream;

        EncryptedFileStreamWriter(const self &) = delete;

        self &operator=(const self &) = delete;

        EncryptedFileStreamWriter() = delete;

        /**
         *
         * @param path file path
         * @param key length over 32 will be ignore
         */
        explicit EncryptedFileStreamWriter(const std::string &path, const std::string &key);

        ~EncryptedFileStreamWriter();

        bool is_open() const;

        void close();

        size_t write(const void *buffer, size_t size) final;

    private:
        class Implement;
        Declare<Implement> m_impl;
    };

}


#endif //TENSORSTACK_ENCRYPTION_ENCRYPTED_FSTREAM_H
