//
// Created by kier on 19-4-25.
//

#ifndef TENNIS_API_CPP_STREAM_H
#define TENNIS_API_CPP_STREAM_H

#include "../stream.h"
#include <fstream>
#include <cstring>

namespace ts {
    namespace api {
        /**
         * @see ts_stream_read
         */
        class StreamReader {
        public:
            using self = StreamReader;

            virtual uint64_t read(void *buf, uint64_t len) = 0;

            static uint64_t C(void *obj, char *buf, uint64_t len) {
                return reinterpret_cast<self *>(obj)->read(buf, len);
            }
        };

        /**
         * @see ts_stream_write
         */
        class StreamWriter {
        public:
            using self = StreamWriter;

            virtual uint64_t write(const void *buf, uint64_t len) = 0;

            static uint64_t C(void *obj, const char *buf, uint64_t len) {
                return reinterpret_cast<self *>(obj)->write(buf, len);
            }
        };


        class FileStreamReader : public StreamReader {
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

            uint64_t read(void *buffer, uint64_t size) final;

            std_stream &stream() { return m_stream; }

            const std_stream &stream() const { return m_stream; }

        private:
            std_stream m_stream;
        };

        class FileStreamWriter : public StreamWriter {
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

            uint64_t write(const void *buffer, uint64_t size) final;

            std_stream &stream() { return m_stream; }

            const std_stream &stream() const { return m_stream; }

        private:
            std_stream m_stream;
        };

        class BufferReader : public StreamReader {
        public:
            using self = BufferReader;
            using supper = StreamReader;
            using std_stream = std::ifstream;

            BufferReader(const self &) = delete;

            self &operator=(const self &) = delete;

            BufferReader(const void *buffer, uint64_t size)
                : m_buffer(reinterpret_cast<const char *>(buffer)), m_size(size) {}

            uint64_t read(void *buffer, uint64_t size) final;

            void reset() { m_index = 0; };

            uint64_t size() const { return m_size; }

            uint64_t index() const { return m_index; }

        private:
            const char *m_buffer = nullptr;
            uint64_t m_size = 0;
            uint64_t m_index = 0;
        };

        bool FileStreamReader::is_open() const {
            return m_stream.is_open();
        }

        uint64_t FileStreamReader::read(void *buffer, uint64_t size) {
            m_stream.read(reinterpret_cast<char *>(buffer), size);
            return uint64_t(m_stream.gcount());
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

        uint64_t FileStreamWriter::write(const void *buffer, uint64_t size) {
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

        uint64_t BufferReader::read(void *buffer, uint64_t size) {
            if (m_buffer == nullptr || m_index >= m_size) return 0;
            auto read_size = std::min(size, m_size - m_index);

            std::memcpy(buffer, m_buffer + m_index, size_t(read_size));

            m_index += read_size;
            return read_size;
        }
    }
}

#endif //TENNIS_API_CPP_STREAM_H
