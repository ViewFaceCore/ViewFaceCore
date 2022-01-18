#include "fast_encryptstream.h"
#include <orz/utils/log.h>
#include <thread>

namespace orz
{
    static void encrypt( unsigned char *buf, int64_t len, int64_t key )
    {
        int64_t nstep = len / 8;
        int64_t *ptr = ( int64_t * )buf;
        for( int64_t i = 0; i < nstep; i++ )
        {
            ptr[i] = ptr[i] ^ key;
        }
    }

    static void decrypt( unsigned char *buf, int64_t len, int64_t key )
    {
        int64_t nstep = len / 8;
        int64_t *ptr = ( int64_t * )buf;
        for( int64_t i = 0; i < nstep; i++ )
        {
            ptr[i] = ptr[i] ^ key;
        }
    }


    int64_t Fast_EncryptInputStream::read( char *buffer, int64_t len )
    {
        if( m_in == nullptr )
        {
            return -1;
        }

        if( len < 0 )
        {
            return -1;
        }
        else
            if( len == 0 )
            {
                return 0;
            }

        if( m_master_datalen - m_master_offset >= len )
        {
            memcpy( buffer, m_master + m_master_offset, len );
            m_master_offset += len;
            return len;
        }

        //bool iseof = false;
        int64_t buffer_offset = 0;
        if( m_master_datalen - m_master_offset > 0 )
        {
            memcpy( buffer, m_master + m_master_offset, m_master_datalen - m_master_offset );
            buffer_offset = m_master_datalen - m_master_offset;
            m_master_offset = 0;
            m_master_datalen = 0;
        }

        if( is_eof() )
        {
            if( buffer_offset > 0 )
            {
                return buffer_offset;
            }
            else
            {
                //orz::Log(orz::ERROR) << "mode file is eof!";
                return 0;
            }
        }

        int64_t nremain = ( len - buffer_offset ) % AES_BLOCKLEN;

        int nreads = m_in->read( buffer + buffer_offset, len - buffer_offset - nremain );
        if( nreads <= 0 )
        {
            m_eof = true;
            return buffer_offset;
        }

        if( nreads % AES_BLOCKLEN != 0 )
        {
            orz::Log( orz::ERROR ) << "mode file read format is error!" << orz::crash;
            return 0;
        }
        if( nreads < len - buffer_offset - nremain )
        {
            m_eof = true;
        }

        decrypt( ( unsigned char * )( buffer + buffer_offset ), nreads, m_cskey );
        buffer_offset += nreads;

        if( m_eof )
        {
            return buffer_offset;
        }

        if( nremain > 0 )
        {
            m_master_datalen = m_in->read( reinterpret_cast<char *>( m_master ), AES_BLOCKLEN );
            if( m_master_datalen <= 0 )
            {
                m_eof = true;
                return buffer_offset;
            }
            else
            {
                if( m_master_datalen != AES_BLOCKLEN )
                {
                    orz::Log( orz::ERROR ) << "mode file read format is error!" << orz::crash;
                    return 0;
                }
                //AES_ECB_decrypt(&m_ctx, m_second);
                decrypt( ( unsigned char * )m_master, AES_BLOCKLEN, m_cskey );
                memcpy( buffer + buffer_offset, m_master, nremain );
                m_master_offset = nremain;
                buffer_offset += nremain;
            }
        }

        return buffer_offset;
    }



    Fast_EncryptInputStream::Fast_EncryptInputStream( std::shared_ptr<InputStream> in, const std::string &key ): FilterInputStream( in )
    {
        m_master_offset = 0;
        m_master_datalen = 0;
        //m_second_datalen = 0;
        m_eof = false;
        m_key = key;
        if( key.length() < 1 )
        {
            orz::Log( orz::DEBUG ) << "Using key is empty"  << orz::crash;
        }

        int64_t nkey = 0;
        for( int i = 0; i < key.length(); i++ )
        {
            nkey *= 10;
            nkey += key[i];
        }
        m_cskey = nkey;
        //int nret = AES_set_decrypt_key((unsigned char *)buf, 128, &m_aeskey);
        //if(nret != 0) {
        //    orz::Log(orz::FATAL) << "AES_set_decrypt_key failed:" << nret << orz::crash;
        //}

    }


    Fast_EncryptInputStream::~Fast_EncryptInputStream()
    {
    }


    /////////////////////////////////////
    int64_t Fast_EncryptOutputStream::write( const char *buffer, int64_t len )
    {
        if( m_out == nullptr )
        {
            return -1;
        }

        int64_t nwrite = 0;
        while( nwrite < len )
        {
            if( AES_BLOCKLEN - m_master_datalen  >= len - nwrite )
            {
                memcpy( m_master + m_master_datalen, reinterpret_cast<const char *>( buffer ) + nwrite, len - nwrite );
                m_master_datalen += len - nwrite;
                nwrite = len;
                return nwrite;
            }
            else
            {
                memcpy( m_master + m_master_datalen, reinterpret_cast<const char *>( buffer ) + nwrite, AES_BLOCKLEN - m_master_datalen );
                //AES_ECB_encrypt(&m_ctx, m_master);
                encrypt( ( unsigned char * )m_master, AES_BLOCKLEN, m_cskey );
                if( m_out->write( reinterpret_cast<const char *>( m_master ), AES_BLOCKLEN ) != AES_BLOCKLEN )
                {
                    orz::Log( orz::ERROR ) << "Fast_EncryptOutputStream write failed!" << orz::crash;
                    return 0;
                }
                nwrite += AES_BLOCKLEN - m_master_datalen;
                m_master_datalen = 0;

            }
        }
        return nwrite;
    }


    void Fast_EncryptOutputStream::flush()
    {
        if( m_out == nullptr )
        {
            return;
        }

        if( m_master_datalen == AES_BLOCKLEN )
        {
            //AES_ECB_encrypt(&m_ctx, m_master);

            encrypt( m_master, AES_BLOCKLEN, m_cskey );
            //encrypt(m_master, AES_BLOCKLEN);
            //AES_ecb_encrypt(m_master, m_master, &m_aeskey, AES_ENCRYPT);
            if( m_out->write( reinterpret_cast<const char *>( m_master ), AES_BLOCKLEN ) != AES_BLOCKLEN )
            {
                orz::Log( orz::ERROR ) << "Fast_EncryptOutputStream write failed!" << orz::crash;
                return;
            }
            m_master_datalen = 0;
        }
        uint8_t npadding = ( uint8_t )( AES_BLOCKLEN - m_master_datalen );
        uint8_t buf[AES_BLOCKLEN];
        memset( buf, npadding, AES_BLOCKLEN );
        memcpy( buf, m_master, m_master_datalen );
        //AES_ECB_encrypt(&m_ctx, buf);

        encrypt( buf, AES_BLOCKLEN, m_cskey );
        //encrypt(buf, AES_BLOCKLEN);
        //AES_ecb_encrypt(buf, buf, &m_aeskey, AES_ENCRYPT);
        if( m_out->write( reinterpret_cast<const char *>( buf ), AES_BLOCKLEN ) != AES_BLOCKLEN )
        {
            orz::Log( orz::ERROR ) << "Fast_EncryptOutputStream write failed!" << orz::crash;
            return;
        }
        m_master_datalen = 0;
    }



    Fast_EncryptOutputStream::Fast_EncryptOutputStream( std::shared_ptr<OutputStream> out, const std::string &key ): FilterOutputStream( out )
    {
        m_master_datalen = 0;
        m_key = key;

        if( key.length() < 1 )
        {
            orz::Log( orz::DEBUG ) << "Using key is empty"  << orz::crash;
        }

        int64_t nkey = 0;
        for( int i = 0; i < key.length(); i++ )
        {
            nkey *= 10;
            nkey += key[i];
        }
        m_cskey = nkey;

    }


    Fast_EncryptOutputStream::~Fast_EncryptOutputStream()
    {
        flush();
    }





}
