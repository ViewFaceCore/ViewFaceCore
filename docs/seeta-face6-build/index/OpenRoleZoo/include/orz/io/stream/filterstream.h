#ifndef ORZ_IO_STREAM_FILTERSTREAM_H
#define ORZ_IO_STREAM_FILTERSTREAM_H

#include "orz/io/stream/stream.h"
#include <memory>

namespace orz {


/**
 * A FilterInputStream contains some other input stream, which it uses as its basic
 * source of data, possibly transforming the data along the way or providing additional functionality.
 */
class FilterInputStream : public InputStream {
public:
    /**
    * reads up to len char of data from this input stream into a char array. 
    * This method blocks until some input is available.
    * @param buf, Pointer to an array where the read characters are stored. 
    * @param len, Number of characters to read.
    * @return return the number of characters read, If no char is available because 
    * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
    */
    int64_t read(char *buf, int64_t len) override;

protected:
    /**
     * Creates a FilterInputStream by assigning the argument in to the field this.m_in so
     * as to remember it for later use
     * @param in the underlying input stream
     */
    FilterInputStream(std::shared_ptr<InputStream> in);


protected:
    /**
     * The input stream to be filtered
     */
    std::shared_ptr<InputStream> m_in;

};

/**
 * This class is the superclass of all classes that filter output streams.
 * These strreams sit on top of an already existing output stream which it uses 
 * as its basic sink of data, buf possibly transforming the data along the way
 * or providing additional functionality. 
 */
class FilterOutputStream : public OutputStream {
public:
    /**
    @param buf, Pointer to an array of at least len characters.. 
    @param len, Number of characters to insert.
    @return return the number of characters written, an exception happen will return -1.
    */
    int64_t write(const char *buf, int64_t len) override;

protected:

    /**
     * Creates a FilterOutputStream by assigning the argument out to the field this.m_out so
     * as to remember it for later use
     * @param in the underlying output stream
     */
    FilterOutputStream(std::shared_ptr<OutputStream> out);


protected:

    /**
     * The output stream to be filtered
     */
    std::shared_ptr<OutputStream> m_out;

};


}



#endif
