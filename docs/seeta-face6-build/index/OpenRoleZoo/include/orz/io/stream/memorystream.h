#ifndef ORZ_IO_STREAM_MEMORYSTREAM_H
#define ORZ_IO_STREAM_MEMORYSTREAM_H

#include "orz/io/stream/stream.h"
#include <memory>
#include <string>

namespace orz {


/**
 * A memoryInputStream contains an internal buffer that contains chars that may be read from stream.
 * An internal counter keeps track of the char to be supplied by the read method
 */

class MemoryInputStream : public InputStream {
public:
    /**
    * @param buf, Pointer to an array where the read characters are stored. 
    * @param len, Number of characters to read.
    * @return return the number of characters read, If no char is available because 
    * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
    */
    int64_t read(char *buf, int64_t len) override;

    /**
     * Create a MemoryInputStream so that it uses buf as its buffer array.
     *@param buf the input buffer
     *@param len the input buffer valid data length
     */ 
    MemoryInputStream(const char *buf, int64_t len);
    /**
     * @param pos set the offset position in the stream.
     * @return return true if set the offset position succeed, otherwise return false
     */
    bool setpos(int64_t pos);

    /**
     * @return return the number of remaining chars that can be read from this input stream.
     */
    int64_t available() { return m_len - m_pos; }
protected:
    /**
     * An array of char that was provided by the creator of the stream. 
     */
    std::shared_ptr<char> m_buf;
    /**
     * The length of the array's data.
     * This value should always be nonnegative.
     */
    int64_t m_len;
    
    /**
     * The index of the next character to read from input stream buffer.
     * This value should always be nonnegative and not larger than the value of m_len
     */
    int64_t m_pos;
};


/**
 * This clas implements an output stream in which the data is written into a char array.
 * The buffer automatically grows as data is written to it.
 */
class MemoryOutputStream : public OutputStream {
public:
    /**
     * @param buf, Pointer to an array of at least len characters.. 
     * @param len, Number of characters to insert.
     * @return return the number of characters written, an exception happen will return -1.
     */
    int64_t write(const char *buf, int64_t len) override;

    /**
     * Create a MemoryOutputStream. The buffer capacity is initially 32 bytes, though its size increases if necessary
     * @param size default buffer size
     */
    MemoryOutputStream(int64_t size = 32);

    /** 
     * @return return the buffer valid data length
     */
    int64_t size() { return m_pos; }
    /**
     * @return return the buffer data
     */
    std::string getdata();

    /**
     * @param pos set the offset position in the stream.
     * @return return true if set the offset position succeed, otherwise return false
     */
    bool setpos(int64_t pos);
private:
    /**
     * @param mincapacity the buffer grow size
     * the buffer size must lesss than std::numberic_limits<int64_t>::max()
     */
    void grow(int64_t mincapacity);
protected:
    /**
     * The buffer where data is stored
     */
    std::shared_ptr<char> m_buf;
    /**
     * The number of capacity the buffer.
     * m_len < std::numberic_limits<int64_t>::max()
     */
    int64_t m_len;

    /**
     * The number of valid chars in the buffer.
     */
    int64_t m_pos;
};


}



#endif
