#ifndef ORZ_IO_STREAM_FILESTREAM_H
#define ORZ_IO_STREAM_FILESTREAM_H

#include "orz/io/stream/stream.h"
#include <fstream>
#include <string>

namespace orz {

/**
 * A FileInputStream obtains input bytes from a file
 */
class FileInputStream : public InputStream {
public:
    /**
    * @param buf, Pointer to an array where the read characters are stored. 
    * @param len, Number of characters to read.
    * @return return the number of characters read. If no char is available because
    * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
    */
    int64_t read(char *buf, int64_t len) override;

    /**
    * @param file, the file's path. 
    * @param istxt, if the file is not binary file, set is true. default handle binary file.
    */
    FileInputStream(const std::string &file, bool istxt=false);

    virtual ~FileInputStream();
    
    /**
     * @return return the current file name
     */
    std::string getfile() { return m_file; }
    /**
     * @return return the current file input stream object
     */
    std::ifstream & getstream() { return m_in; }
    /**
    @return return the inputstream work path.
    */
    virtual const char * getworkpath(); 

private:
    /**
     * the underlying file input stream
     */
    std::ifstream m_in;
    /**
     * the file name
     */
    std::string m_file;

    /**
     * the file path 
     */
    std::string m_workpath;
};


/**
 * A FileOutputStream is an output stream for writing data to a file 
 */
class FileOutputStream : public OutputStream {
public:
    /**
    @param buf, Pointer to an array of at least len characters.. 
    @param len, Number of characters to insert.
    @return return the number of characters written. an exception happen will return -1.
    */
    int64_t write(const char *buf, int64_t len) override;

    /**
    * @param file, the file's path. 
    * @param istxt, if the file is not binary file, set is true. default handle binary file.
    */
    FileOutputStream(const std::string &file, bool istxt=false);
    virtual ~FileOutputStream();

    /**
     * @return return the current file name
     */
    std::string getfile() { return m_file; }

    /**
     * @return return the current file output stream object
     */
    std::ofstream & getstream() { return m_out; }

    /**
    @return return the outputstream work path.
    */
    virtual const char * getworkpath(); 

private:

    /**
     * the underlying file output stream
     */
    std::ofstream m_out;

    /**
     * the file name
     */
    std::string m_file;
    
    /**
     * the file path 
     */
    std::string m_workpath;

};


}



#endif
