//
// Created by lby on 2018/1/15.
//

#ifndef ORZ_IO_I_H
#define ORZ_IO_I_H

#include "jug/binary.h"

#include <istream>

namespace orz {
    binary read_file(const std::string &filename);

    std::string read_txt_file(const std::string &filename);

    class imemorystream : public std::istream {
    public:
        imemorystream(const void *data, size_t size);

    private:
        class imemorybuffer : public std::streambuf {

        public:
            imemorybuffer(const void *data, size_t size);

        protected:
            int_type overflow(int_type c) override;

            std::streambuf *setbuf(char *s, std::streamsize n) override;

            int_type underflow() override;

            int_type uflow() override;

            streampos seekoff(streamoff off, ios_base::seekdir way,
                              ios_base::openmode which) override;

            streampos seekpos(streampos sp, ios_base::openmode which) override;

        private:
            const void *m_data;
            size_t m_size;
        };

        imemorybuffer m_buffer;
    };

    class MemoryFILE {
    public:
        using self = MemoryFILE;

        explicit MemoryFILE(::FILE *file);

        MemoryFILE(void *data, size_t size);

        friend size_t fread(void *ptr, size_t size, size_t count, MemoryFILE *stream);

        friend size_t fwrite(const void *ptr, size_t size, size_t count, MemoryFILE *stream);

        friend int fseek(MemoryFILE *stream, long int offset, int origin);

        friend long int ftell(MemoryFILE *stream);

        friend void rewind(MemoryFILE *stream);

        friend int ferror(MemoryFILE *stream);

        friend int feof(MemoryFILE *stream);

    private:
        ::FILE *m_file = nullptr;
        char *m_data = nullptr;
        size_t m_size = 0;
        size_t m_index = 0;
    };

    using FILE = MemoryFILE;
}

#endif //ORZ_IO_I_H
