#include "cstastream.h"
#include "orz/utils/log.h"


namespace orz
{

    CstaInputStream::CstaInputStream( std::shared_ptr<InputStream> in ):
        FilterInputStream( in )
    {

        unsigned int mask = 0;

        int64_t nreads = m_in->read( ( char * )&mask, 4 );
        if( ( nreads != 4 ) || ( mask != 0x61746573 ) )
        {
            orz::Log( orz::ERROR ) << "the csta InputStream is invalid!" << orz::crash;
        }

    }



    //////////////////////////////////////////////
    CstaOutputStream::CstaOutputStream( std::shared_ptr<OutputStream> out )
        : FilterOutputStream( out )
    {
        unsigned int mask = 0x61746573;
        int64_t nwrites = m_out->write( ( const char * )&mask, 4 );
        if( nwrites != 4 )
        {
            orz::Log( orz::ERROR ) << "wirte csta OutputStream failed!" << orz::crash;
        }
    }


}
