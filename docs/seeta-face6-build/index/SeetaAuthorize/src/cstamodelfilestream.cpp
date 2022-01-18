#include "cstamodelfilestream.h"


#include <orz/utils/log.h>
#include "orz/io/stream/filestream.h"
#include "orz/io/stream/memorystream.h"
#include "cstastream.h"
//#include "encryptstream.h"

#include "fast_cstastream.h"
#include "fast_encryptstream.h"


//#include "easy_aes.h"
#include "orz/codec/json.h"
#include "orz/io/i.h"

#include "orz/io/dir.h"
#include <fstream>

namespace orz
{

    int64_t CstaModelFileInputStream::read( char *buf, int64_t len )
    {
        return m_in->read( buf, len );
    }

    int CstaModelFileInputStream::get_encrypt_type()
    {
        return m_encrypt_type;
    }


    CstaModelFileInputStream::CstaModelFileInputStream( const std::string &file, const std::string &key )
    {
        unsigned int mask = 0;
        int64_t nreads = 0;
        m_size = 0;
        m_istxt = false;
        m_file = file;
        m_key = key;
        m_encrypt_type = -1;
        {
            std::ifstream reader( file.c_str(), std::ios_base::binary );
            if( !reader.is_open() )
            {
                orz::Log( orz::ERROR ) << "open the model file:" << file << " failed!" << orz::crash;
            }
            reader.read( ( char * )&mask, 4 );
            if( reader.bad() )
            {
                orz::Log( orz::ERROR ) << "read the model file:" << file << " failed!" << orz::crash;
            }
            reader.seekg( 0, std::ios::end );
            m_size = reader.tellg();

            reader.close();
        }

        if( mask == 0x19910929 )
        {
            m_in = std::shared_ptr<InputStream>( new FileInputStream( file ) );
        }

        else
            if( mask == 0x61746573 )
            {
 
                orz::Log( orz::ERROR ) << "model file:" << file << " is not supported!" << orz::crash;
                /*
                if( key.length() < 1 )
                {
                    orz::Log( orz::ERROR ) << "paramter key is empty!" << orz::crash;
                }

                m_encrypt_type = 1;
                std::shared_ptr<InputStream> csta_file = std::shared_ptr<InputStream>( new FileInputStream( file ) );
                std::shared_ptr<InputStream> csta = std::shared_ptr<InputStream>( new CstaInputStream( csta_file ) );
                m_in = std::shared_ptr<InputStream>( new EncryptInputStream( csta, key ) );
                */
            }

            else
                if( mask == 0x74736166 )
                {
                    if( key.length() < 1 )
                    {
                        orz::Log( orz::ERROR ) << "paramter key is empty!" << orz::crash;
                    }

                    m_encrypt_type = 0;
                    std::shared_ptr<InputStream> csta_file = std::shared_ptr<InputStream>( new FileInputStream( file ) );
                    std::shared_ptr<InputStream> csta = std::shared_ptr<InputStream>( new Fast_CstaInputStream( csta_file ) );
                    m_in = std::shared_ptr<InputStream>( new Fast_EncryptInputStream( csta, key ) );

                }
                else
                {
                    m_istxt = true;
                    m_in = std::shared_ptr<InputStream>( new FileInputStream( file, true ) );
                }

    }

    CstaModelFileInputStream::~CstaModelFileInputStream()
    {
    }

    orz::jug CstaModelFileInputStream::read_jug()
    {
        std::shared_ptr<char> buf = std::shared_ptr<char>( new char[m_size], std::default_delete<char []>() );
        int64_t nreads = read( buf.get(), m_size );
        if( nreads <= 0 )
        {
            orz::Log( orz::ERROR ) << "read model file: " << m_file << " failed!" << orz::crash;
            return orz::jug();
        }
        if( m_istxt )
        {
            std::string str( buf.get(), nreads );

            std::string strpath;
            std::string strsep = "/"; //orz::FileSeparator();
            int nfind = m_file.find_last_of( strsep );
            if( nfind >= 0 )
            {
                strpath = m_file.substr( 0, nfind + 1 );
            }
            else
            {
                strsep = "\\";
                nfind = m_file.find_last_of( strsep );
                if( nfind >= 0 )
                {
                    strpath = m_file.substr( 0, nfind + 1 );
                }
            }
            return json2jug( str, strpath );
        }
        else
        {
            orz::imemorystream in( buf.get() + 4, nreads - 4 );
            auto model  = orz::jug_read( in );
            return model;
        }
    }


    orz::jug CstaModelFileInputStream::read_jug( std::shared_ptr<InputStream> in, const std::string &key )
    {
        unsigned int mask = 0;
        int64_t nreads = 0;
        bool istxt = false;
        int noffset = 0;

        nreads = in->read( ( char * )&mask, 4 );
        if( nreads != 4 )
        {
            orz::Log( orz::ERROR ) << "InputStream format is error!" << orz::crash;
        }

        std::shared_ptr<InputStream> ifile;
        if( mask == 0x19910929 )
        {
            ifile = in;
        }

        else
            if( mask == 0x61746573 )
            {

                orz::Log( orz::ERROR ) << "InputStream format is error!" << orz::crash;
                /*
                if( key.length() < 1 )
                {
                    orz::Log( orz::ERROR ) << "paramter key is empty!" << orz::crash;
                }
                noffset = 4;
                ifile = std::shared_ptr<InputStream>( new EncryptInputStream( in, key ) );
                */
            }

            else
                if( mask == 0x74736166 )
                {
                    if( key.length() < 1 )
                    {
                        orz::Log( orz::ERROR ) << "paramter key is empty!" << orz::crash;
                    }
                    noffset = 4;
                    ifile = std::shared_ptr<InputStream>( new Fast_EncryptInputStream( in, key ) );

                }
                else
                {
                    istxt = true;
                    ifile = in;
                }

        char buf[10240];
        int len = sizeof( buf );

        orz::MemoryOutputStream omemory( 10240000 );

        if( istxt )
        {
            omemory.write( ( const char * )&mask, 4 );
        }

        while( 1 )
        {
            nreads = ifile->read( buf, len );
            if( nreads > 0 )
            {
                if( omemory.write( buf, nreads ) != nreads )
                {
                    orz::Log( orz::ERROR ) << "write memoryoutputstream failed!" << orz::crash;
                }
            }
            else
            {
                break;
            }
        }

        std::string data = omemory.getdata();
        if( istxt )
        {
            std::string strpath = in->getworkpath();
            return json2jug( data, strpath );
        }
        else
        {
            orz::imemorystream in( data.data() + noffset, data.length() - noffset );
            auto model  = orz::jug_read( in );
            return model;
        }
        return orz::jug();
    }



    /////////////////////////////////////
    int64_t CstaModelFileOutputStream::write( const char *buf, int64_t len )
    {
        int64_t nwrite = -1;
        nwrite = m_out->write( buf, len );
        return nwrite;
    }

    int CstaModelFileOutputStream::get_encrypt_type()
    {
        return m_encrypt_type;
    }


    CstaModelFileOutputStream::CstaModelFileOutputStream( const std::string &file, bool istxt, const std::string &key, int encrypt_type )
    {
        m_key = key;
        m_istxt = istxt;
        m_encrypt_type = encrypt_type;
        if( m_istxt )
        {
            m_out = std::shared_ptr<OutputStream>( new FileOutputStream( file, true ) );
        }
        else
        {
            if( key.length() < 1 )
            {
                m_out = std::shared_ptr<OutputStream>( new FileOutputStream( file ) );
            }
            else
            {
                if( encrypt_type == 0 )
                {
                    std::shared_ptr<OutputStream> csta_file = std::shared_ptr<OutputStream>( new FileOutputStream( file ) );
                    std::shared_ptr<OutputStream> csta = std::shared_ptr<OutputStream>( new Fast_CstaOutputStream( csta_file ) );
                    m_out = std::shared_ptr<OutputStream>( new Fast_EncryptOutputStream( csta, key ) );
                    //std::cout << "fast encrypt_type:" << encrypt_type << std::endl;
                }
/*
                else
                    if( encrypt_type == 1 )
                    {
                        std::shared_ptr<OutputStream> csta_file = std::shared_ptr<OutputStream>( new FileOutputStream( file ) );
                        std::shared_ptr<OutputStream> csta = std::shared_ptr<OutputStream>( new CstaOutputStream( csta_file ) );
                        m_out = std::shared_ptr<OutputStream>( new EncryptOutputStream( csta, key ) );

                        //std::cout << "encrypt_type:" << encrypt_type << std::endl;
                    }
*/
                    else
                    {
                        orz::Log( orz::FATAL ) << "not support encrypt type:" << encrypt_type << orz::crash;
                    }

            }
        }
    }


    CstaModelFileOutputStream::~CstaModelFileOutputStream()
    {
    }





}
