#ifndef ORZ_IO_STREAM_STREAM_H
#define ORZ_IO_STREAM_STREAM_H

#include <stdint.h>

namespace orz {

/**
 *This abstract class is the superclass of all classes representing an input stream of bytes.
 */

class InputStream {
public:

    /**
     @param buf, Pointer to an array where the extracted characters are stored. 
     @param len, Number of characters to extract.
     @return return the number of characters extracted.
    */
    virtual int64_t read(char *buf, int64_t len) = 0;
    virtual ~InputStream() {}

    /**
     @return return the inputstream work path.
    */
    virtual const char * getworkpath() { return ""; }
};


class OutputStream {
public:

    /**
     @param buf, Pointer to an array of at least len characters.. 
     @param len, Number of characters to insert.
     @return return the number of characters inserted.
    */
    virtual int64_t write(const char *buf, int64_t len) = 0;
    virtual ~OutputStream() {}

    /**
      @return return the outputstream work path.
    */
    virtual const char * getworkpath() { return ""; }
};


}


#endif
