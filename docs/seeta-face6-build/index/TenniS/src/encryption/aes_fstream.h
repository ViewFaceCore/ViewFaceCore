//
// Created by wqy on 2019/02/25.
//

#ifndef TENSORSTACK_ENCRYPTION_AES_FSTREAM_H
#define TENSORSTACK_ENCRYPTION_AES_FSTREAM_H

#include "module/io/stream.h"
#include <fstream>
#include <vector>
#include <memory>
#include "aes.h"

namespace ts {
    class AESFileStreamReader : public StreamReader {
    public:
        using self = AESFileStreamReader;
        using supper = StreamReader;
        using std_stream = std::ifstream;

        AESFileStreamReader(const self &) = delete;

        self &operator=(const self &) = delete;

        AESFileStreamReader() = delete;

        explicit AESFileStreamReader(const std::string &path, const std::string &key);

        ~AESFileStreamReader();

        //void open(const std::string &path);

        bool is_open() const;

        void close();

        size_t read(void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
        uint8_t m_master[AES_BLOCKLEN];
        uint8_t m_second[AES_BLOCKLEN];

        int        m_master_datalen;
        int        m_master_offset;

        int        m_second_datalen;
        struct AES_ctx m_ctx;
    };


    class AESFileStreamWriter : public StreamWriter {
    public:
        using self = AESFileStreamWriter;
        using supper = StreamWriter;
        using std_stream = std::ofstream;

        AESFileStreamWriter(const self &) = delete;

        self &operator=(const self &) = delete;

        AESFileStreamWriter() = delete;

        explicit AESFileStreamWriter(const std::string &path, const std::string &key);

        ~AESFileStreamWriter();
        //void open(const std::string &path);

        bool is_open() const;

        void close();

        size_t write(const void *buffer, size_t size) final;

        std_stream &stream() { return m_stream; }

        const std_stream &stream() const { return m_stream; }

    private:
        std_stream m_stream;
        uint8_t m_master[AES_BLOCKLEN];

        int        m_master_datalen;

        struct AES_ctx m_ctx;
    };

}
#endif //TENSORSTACK_ENCRYPTION_AES_FSTREAM_H
