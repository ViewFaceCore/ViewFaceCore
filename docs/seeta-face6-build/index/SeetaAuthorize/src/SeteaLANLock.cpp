#include "SeetaLANLock.h"
#include "SeetaLockFunction.h"
#include <string>
#include <mutex>

#include "cstamodelfilestream.h"


#include <orz/net/http.h>
#include <orz/utils/format.h>
#include <orz/utils/random.h>
#include <orz/io/jug/jug.h>
#include <orz/codec/json.h>
//#include <orz/ssl/aes.h>
#include <orz/codec/base64.h>
#include <orz/tools/timer.h>
#include <queue>

#include "error_code.h"

#if _MSC_VER>=1900
#include "stdio.h"
#endif

orz::jug GetModelJug( const char *file )
{
    std::string strfile( file );
        const std::string key = "seetatech.com";
    orz::CstaModelFileInputStream stream( strfile, key );
    orz::jug model = stream.read_jug();
        std::string str;

    if( !model.valid( orz::Piece::DICT ) )
    {
            //orz::Log(orz::FATAL) << "read model file failed!" << orz::crash;
        str = error_str( 10001 );
            str += ":";
            str += strfile;
        orz::Log( orz::FATAL ) << str << orz::crash;
    }
    return model;
}

orz::jug GetModelJug( orz::InputStream *in )
{
        const std::string key = "seetatech.com";
    auto stream = std::shared_ptr<orz::InputStream>( in, []( orz::InputStream * p ) { } );
    orz::jug model = orz::CstaModelFileInputStream::read_jug( stream, key );
        std::string str;

    if( !model.valid( orz::Piece::DICT ) )
    {
            //orz::Log(orz::FATAL) << "read model file failed!" << orz::crash;
        str = error_str( 10001 );
        orz::Log( orz::FATAL ) << str << orz::crash;
    }
    return model;
}

void SeetaLock_call( SeetaLock_Function *function )
{
    switch( function->id )
    {
    case SeetaLock_Verify_GetModelJug_ID:
    {
            function->serial_number = next_serial_number( function->serial_number );
            auto core = static_cast<SeetaLock_Verify_GetModelJug *>( function );
            core->out.modeljug = GetModelJug( core->in.file );
        core->out.errcode = 0;
    }
    break;
    case SeetaLock_Verify_GetModelJug_FromStream_ID:
    {
            function->serial_number = next_serial_number( function->serial_number );
            auto core = static_cast<SeetaLock_Verify_GetModelJug_FromStream *>( function );
            core->out.modeljug = GetModelJug( core->in.stream );
        core->out.errcode = 0;
    }
    default:
        break;
    }
}
