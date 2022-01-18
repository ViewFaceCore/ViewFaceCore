//
// Created by lby on 2018/1/15.
//

#include "orz/io/i.h"
#include <fstream>
#include <sstream>

#include <cstring>
#include <algorithm>

namespace orz {
    binary read_file(const std::string &filename) {
        binary bin;
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return bin;
        in.seekg(0, std::ios::end);
        auto size = in.tellg();
        bin.resize(size_t(size));
        in.seekg(0, std::ios::beg);
        in.read(bin.data<char>(), bin.size());
        in.close();
        return bin;
    }

	std::string read_txt_file(const std::string& filename)
	{
		std::ifstream in(filename);
		std::ostringstream tmp;
		tmp << in.rdbuf();
		return tmp.str();
	}

    imemorystream::imemorystream(const void *data, size_t size) :
            std::istream(&m_buffer),
            m_buffer(data, size) {
    }

    imemorystream::imemorybuffer::imemorybuffer(const void *data, size_t size) :
            m_data(data), m_size(size) {
        setbuf((char *) data, size);
    }

    int imemorystream::imemorybuffer::overflow(int c) {
        return c;
    }

    std::streambuf *imemorystream::imemorybuffer::setbuf(char *s, std::streamsize n) {
        // setp(s, s + n);  // do not set output pointer
        setg(s, s, s + n);
        return this;
    }

    int imemorystream::imemorybuffer::underflow() {
        return EOF;
    }

    int imemorystream::imemorybuffer::uflow() {
        return EOF;
    }

    std::streampos
    imemorystream::imemorybuffer::seekoff(std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode which) {
        if (!(which | ios_base::in)) return gptr() - eback();
        // TODO: check which for omemorystream
        auto pos = gptr();
        switch (way) {
            case ios_base::beg:
                pos = eback() + off;
                break;
            case ios_base::cur:
                pos = gptr() + off;
                break;
            case ios_base::end:
                pos = egptr() + off;
                break;
            default:
                pos = gptr();
                break;
        }
        setg(eback(), pos, egptr());
        return gptr() - eback();
    }

    std::streampos imemorystream::imemorybuffer::seekpos(std::streampos sp, std::ios_base::openmode which) {
        if (!(which | ios_base::in)) return gptr() - eback();
        // TODO: check which for omemorystream
        auto pos = eback() + sp;
        setg(eback(), pos, egptr());
        return sp;
    }

    MemoryFILE::MemoryFILE(::FILE *file) :
        m_file(file) {
    }

    MemoryFILE::MemoryFILE(void *data, size_t size) :
        m_data(reinterpret_cast<char *>(data)), m_size(size){
    }

    size_t fread(void *ptr, size_t size, size_t count, MemoryFILE *stream) {
        if (stream->m_file) return fread(ptr, size, count, stream->m_file);
        size_t read_count =count;
        size_t left_count = (stream->m_size - stream->m_index) / size;
        read_count = std::min(read_count, left_count);
        std::memcpy(ptr, stream->m_data + stream->m_index, read_count * size);
        stream->m_index += read_count * size;
        return read_count;
    }

    size_t fwrite(const void *ptr, size_t size, size_t count, MemoryFILE *stream) {
        if (stream->m_file) return fwrite(ptr, size, count, stream->m_file);
        size_t written_count = count;
        size_t left_count = (stream->m_size - stream->m_index) / size;
        written_count = std::min(written_count, left_count);
        std::memcpy(stream->m_data + stream->m_index, ptr, written_count * size);
        stream->m_index += written_count * size;
        return written_count;
    }

    int fseek(MemoryFILE *stream, long int offset, int origin) {
        if (stream->m_file) return fseek(stream->m_file, offset, origin);
        size_t new_index = stream->m_index;
        switch (origin) {
            case SEEK_SET:
                new_index = static_cast<size_t>(offset);
                break;
            case SEEK_CUR:
                new_index = new_index + offset;
                break;
            case SEEK_END:
                new_index = stream->m_size + offset;
                break;
            default:
                break;
        }
        stream->m_index = new_index;
        return 0;
    }

    long int ftell(MemoryFILE *stream) {
        if (stream->m_file) return ftell(stream->m_file);
        return static_cast<long int>(stream->m_index);
    }

    void rewind(MemoryFILE *stream) {
        fseek(stream, 0, SEEK_SET);
    }

    int ferror(MemoryFILE *stream) {
        if (stream->m_file) return ferror(stream->m_file);
        return stream->m_index >= stream->m_size;
    }

    int feof(MemoryFILE *stream) {
        if (stream->m_file) return feof(stream->m_file);
        return stream->m_index >= stream->m_size;
    }
}
