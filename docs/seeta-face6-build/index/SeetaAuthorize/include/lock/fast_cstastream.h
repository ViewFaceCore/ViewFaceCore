#ifndef ORZ_FAST_CSTASTREAM_H
#define ORZ_FAST_CSTASTREAM_H

#include <memory>
#include <string>
#include "orz/io/stream/filterstream.h"

namespace orz
{

    /**
     * This class transform csta type to sta type
     */
    class Fast_CstaInputStream : public FilterInputStream {
    public:
        /**
         * @param in the input stream that contain csta model data
         * Fast_CstaInputStream skip four bytes mask at the input stream header
         */
        Fast_CstaInputStream( std::shared_ptr<InputStream> in );
    };

    /**
     * This class transform sta type to csta type
     */
    class Fast_CstaOutputStream : public FilterOutputStream {
    public:

        /**
         * @param out the output stream that contain sta model data
         * Fast_CstaOutputStream add four bytes mask at the output stream header
         */
        Fast_CstaOutputStream( std::shared_ptr<OutputStream> out );
    };


}
#endif
