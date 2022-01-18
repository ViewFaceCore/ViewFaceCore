#ifndef ORZ_IO_STREAM_BUFFEREDSTREAM_H
#define ORZ_IO_STREAM_BUFFEREDSTREAM_H

#include "orz/io/stream/filterstream.h"
#include <memory>

namespace orz {

/**
 * the ability to buffer the input stream
 */
class BufferedInputStream : public FilterInputStream {
public:
    /**
    * @param buf, Pointer to an array where the read characters are stored. 
    * @param len, Number of characters to read.
    * @return return the number of characters read, If no char is available because 
    * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
    */
    int64_t read(char *buf, int64_t len) override;
public:
  
    /**
    * @param in, the underlying input stream
    * @param size, the buffer size
    */ 
    BufferedInputStream(std::shared_ptr<InputStream> in, int64_t size = 8192);
private:
    /**
     * The internal buffer array where the data is stored. the buffer's length is m_len.
     * the data's length is m_datalen
     */
    std::shared_ptr<char> m_buf;
    /**
     * the calloced buffer array's length
     */
    int64_t m_len;
    /**
     * the data length of the buffer array
     */ 
    int64_t m_datalen;
     
    /**
     * The current position in the buffer. 
     * This is the index of the next character to be read from  the buffer array. 
     */
    int64_t m_pos;
};


/**
 * the ability to buffer the output stream
 */
class BufferedOutputStream : public FilterOutputStream {
private:
    /**
     * Flushes this buffered output stream.
     * This forces any buffered output bytes to be written out to the underlying output stream. 
     */
    void flush();
public:
    /**
    @param buf, Pointer to an array of at least len characters.. 
    @param len, Number of characters to insert.
    @return return the number of characters written, an exception happen will return -1
    */
    int64_t write(const char *buf, int64_t len) override;

    /**
    * @param out, the underlying output stream
    * @param size, the buffer size
    */ 
    BufferedOutputStream(std::shared_ptr<OutputStream> out, int64_t size=8192);

    /**
     * will to call flush to flush the buffer to outputstream
     */
    ~BufferedOutputStream();
private:

    /**
     * The internal buffer array where the data is stored. the buffer's length is m_len.
     * the data's length is m_datalen
     */
    std::shared_ptr<char> m_buf;

    /**
     * the buffer array's length
     */
    int64_t m_len;

    /**
     * The current position in the buffer. 
     * This is the index of the next character to be write from  the buffer array. 
     */
    int64_t m_pos;
};


}



#endif
