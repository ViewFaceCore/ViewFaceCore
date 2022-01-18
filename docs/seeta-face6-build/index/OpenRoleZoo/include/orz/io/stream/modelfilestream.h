#ifndef ORZ_IO_STREAM_MODELFILESTREAM_H
#define ORZ_IO_STREAM_MODELFILESTREAM_H

#include <string>
#include "orz/io/stream/stream.h"
#include "orz/io/jug/jug.h"


namespace orz {
/**
 * This class parse the json model file and sta model file to get the model file's orz::jug object
*/
class ModelFileInputStream : public InputStream {
public:
    /**
     * @param file the model file path, support json and sta type.
     */
    ModelFileInputStream(const std::string &file);

    ~ModelFileInputStream();
    /**
    * reads up to len char of data from this input stream into a char array. 
    * This method blocks until some input is available.
    * @param buf, Pointer to an array where the read characters are stored. 
    * @param len, Number of characters to read.
    * @return return the number of characters read, If no char is available because 
    * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
    */
    int64_t read(char *buf, int64_t len) override;

    /**
     * parse model file, then transform to orz::jug object 
     * @return return the model's jug object
     */
    orz::jug read_jug();
   
    /**
     * @param in the input stream's contents must is sta format model stream.
     * @return return the model's jug object
     */
    static orz::jug read_jug(std::shared_ptr<InputStream> in);
private:
    /**
     * the unberlying input stream
     */
    std::shared_ptr<InputStream> m_in;
    /**
     * the model file size;
     */
    int64_t m_size;
    /**
     * the model file whether is the json file 
     */
    bool m_istxt;
    /**
     * the model file name
     */
    std::string m_file;
};

/**
 * create model file, may is json or sta format
 */
class ModelFileOutputStream : public OutputStream {
public:
    /**
     * @param file the model file path
     * @param istxt whether is json format
     */
    ModelFileOutputStream(const std::string &file, bool istxt = false);

    ~ModelFileOutputStream();
    /**
    @param buf, Pointer to an array of at least len characters.. 
    @param len, Number of characters to insert.
    @return return the number of characters written, an exception happen will return -1.
    */
    int64_t write(const char *buf, int64_t len) override;
private:

    /**
     * the unberlying output stream
     */
    std::shared_ptr<OutputStream> m_out;

    /**
     * the model file whether is the json file 
     */
    bool m_istxt;

    /**
     * the model file name
     */
    std::string m_file;
};


}

#endif
