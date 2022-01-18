#ifndef ORZ_CSTAMODELFILESTREAM_H
#define ORZ_CSTAMODELFILESTREAM_H

#include <string>
#include "orz/io/stream/stream.h"
#include "orz/io/jug/jug.h"


namespace orz
{


    /**
     * This class parse the json model file,sta and csta model file to get the model file's orz::jug object
    */
    class CstaModelFileInputStream : public InputStream {
    public:
        /**
         * @param file the model file path, support json, sta and csta type.
         * @param key if model file is csta type, the key is decrypt key
         */
        CstaModelFileInputStream( const std::string &file, const std::string &key = "" );

        ~CstaModelFileInputStream();

        /**
        * reads up to len char of data from this input stream into a char array.
        * This method blocks until some input is available.
        * @param buf, Pointer to an array where the read characters are stored.
        * @param len, Number of characters to read.
        * @return return the number of characters read, If no char is available because
        * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
        */
        int64_t read( char *buf, int64_t len ) override;

        /**
         * @return encrypt type, 0: fast encrypt, 1: aes encrypt
         */
        int get_encrypt_type();
        /**
         * parse model file, then transform to orz::jug object
         * @return return the model's jug object
         */
        orz::jug read_jug();

        /**
         * @param in the input stream. supported json, sta, csta model type
         * @return return the model's jug object
         */
        static orz::jug read_jug( std::shared_ptr<InputStream> in, const std::string &key = "" );
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
        /**
         * the decrypt model file key
         */
        std::string m_key;

        /**
         * encrypt type, 0: fast encrypt, 1: aes encrypt
         */
        int m_encrypt_type;
    };

    /**
     * create model file, may is json ,sta or csta format
     */
    class CstaModelFileOutputStream : public OutputStream {
    public:

        /**
         * @param file the model file path
         * @param istxt whether is json format
         * @param key encrypt model file key
         * @param encrypt_type encrypt type
         */
        CstaModelFileOutputStream( const std::string &file, bool istxt = false, const std::string &key = "", int encrypt_type = 0 );

        /**
         * @return encrypt type, 0: fast encrypt, 1: aes encrypt
         */
        int get_encrypt_type();
        ~CstaModelFileOutputStream();
        /**
        @param buf, Pointer to an array of at least len characters..
        @param len, Number of characters to insert.
        @return return the number of characters written, an exception happen will return -1.
        */
        int64_t write( const char *buf, int64_t len ) override;
    private:
        /**
         * the unberlying output stream
         */
        std::shared_ptr<OutputStream> m_out;

        /**
         * the encrypt model file key
         */
        std::string m_key;
        /**
         * the model file whether is the json file
         */
        bool m_istxt;
        /**
         * the model file name
         */
        std::string m_file;

        /**
         * encrypt type, 0: fast encrypt, 1: aes encrypt
         */
        int m_encrypt_type;
    };


}

#endif
