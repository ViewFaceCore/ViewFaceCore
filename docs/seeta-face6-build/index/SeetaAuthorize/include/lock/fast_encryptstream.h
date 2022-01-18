//
// Created by wqy on 2019/05/8.
//

#ifndef ORZ_STREAM_FAST_ENCRYPTSTREAM_H
#define ORZ_STREAM_FAST_ENCRYPTSTREAM_H

#include "orz/io/stream/filterstream.h"
#include <iostream>
#include <vector>
#include <memory>
//#include "easy_aes.h"
#define AES_BLOCKLEN 16
namespace orz
{

    /**
     * The Fast_EncryptInputStream transformat encrypt data to plain data
     * decrypt algorithm is aes
     */
    class Fast_EncryptInputStream : public FilterInputStream {
    public:
        using self = Fast_EncryptInputStream;

        Fast_EncryptInputStream( const self & ) = delete;

        self &operator=( const self & ) = delete;

        Fast_EncryptInputStream() = delete;

        /**
         * @param in the underlying input stream.
         * @param key the decrypt key
         */
        explicit Fast_EncryptInputStream( std::shared_ptr<InputStream> in, const std::string &key );

        virtual ~Fast_EncryptInputStream();

        /**
         * @param buf, Pointer to an array where the read characters are stored.
         * @param len, Number of characters to read.
         * @return return the number of characters read, If no char is available because
         * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
         */
        int64_t read( char *buffer, int64_t len ) override;

        /**
         * @return when the end of input stream has been reached, return true.
         */
        bool is_eof() {
            return m_eof;
        }
    private:
        /**
         * the master buffer, have been used at decrypt
         */
        uint8_t m_master[AES_BLOCKLEN];

        /**
         * the second buffer, have been used at decrypt
         */
        //uint8_t m_second[AES_BLOCKLEN];
        /**
         * the master buffer available data length
         */
        int        m_master_datalen;
        /**
         * the master buffer current reading position
         */
        int        m_master_offset;

        /**
         * the second buffer available data length
         */
        //int        m_second_datalen;

        /**
         * the aes decrypt handle
         */
        //struct AES_ctx m_ctx;

        /**
         * the underlying input stream whether reached the end
         */
        bool       m_eof;

        /**
         * the input stream decrypt key
         */
        std::string m_key;

        //encrypt key hash coce
        int64_t m_cskey;
    };


    class Fast_EncryptOutputStream : public FilterOutputStream {
    public:
        using self = Fast_EncryptOutputStream;

        Fast_EncryptOutputStream( const self & ) = delete;

        self &operator=( const self & ) = delete;

        Fast_EncryptOutputStream() = delete;

        /**
         * @param in the underlying output stream.
         * @param key the encrypt key
         */
        explicit Fast_EncryptOutputStream( std::shared_ptr<OutputStream> out, const std::string &key );

        virtual ~Fast_EncryptOutputStream();

        /**
         * @param buffer, Pointer to an array where the write characters are stored.
         * @param len, Number of characters to write.
         * @return return the number of characters write,
         * an exception happen will return <= 0.
         */
        int64_t write( const char *buffer, int64_t len ) override;

        /**
         * write buffer data to underlying output stream.
         */
        void flush();


    private:
        uint8_t m_master[AES_BLOCKLEN];

        int        m_master_datalen;

        //struct AES_ctx m_ctx;
        //encrypot key
        std::string m_key;

        //encrypt key hash coce
        int64_t m_cskey;
    };


}

#endif
