#ifndef ORZ_CSTASTREAM_H
#define ORZ_CSTASTREAM_H

#include <memory>
#include <string>
#include "orz/io/stream/filterstream.h"

namespace orz
{

    /**
     * This class transform csta type to sta type
     */
    class CstaInputStream : public FilterInputStream {
    public:
        /**
         * @param in the input stream that contain csta model data
         * CstaInputStream skip four bytes mask at the input stream header
         */
        CstaInputStream( std::shared_ptr<InputStream> in );
    };

    /**
     * This class transform sta type to csta type
     */
    class CstaOutputStream : public FilterOutputStream {
    public:

        /**
         * @param out the output stream that contain sta model data
         * CstaOutputStream add four bytes mask at the output stream header
         */
        CstaOutputStream( std::shared_ptr<OutputStream> out );
    };


}
#endif
