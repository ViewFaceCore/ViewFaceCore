//
// Created by kier on 2018/11/7.
//

#ifndef TENSORSTACK_MODULE_SERIALIZATION_H
#define TENSORSTACK_MODULE_SERIALIZATION_H

#include "io/stream.h"

namespace ts {
    /**
     * Interface of serialize and externalize
     * @note Please call ctx::bind<MemoryController> before externalize
     */
    class TS_DEBUG_API Serializable {
    public:
        using self = Serializable;

        virtual ~Serializable() = default;

        /**
         * serialize object to stream
         * @param stream ready stream
         * @return return writen bytes
         */
        virtual size_t serialize(StreamWriter &stream) const = 0;

        /**
         * externalize object from stream
         * @param stream ready stream
         * @return return read bytes
         */
        virtual size_t externalize(StreamReader &stream) = 0;
    };

    /**
     * Interface of serialize and externalize
     * @note Please call ctx::bind<MemoryController> before externalize
     */
    class TS_DEBUG_API SerializableText {
    public:
        using self = SerializableText;

        virtual ~SerializableText() = default;

        /**
         * serialize object to stream
         * @param stream ready stream
         * @return return writen bytes
         */
        virtual size_t serialize_text(StreamWriter &stream) const = 0;

        /**
         * externalize object from stream
         * @param stream ready stream
         * @return return read bytes
         */
        virtual size_t externalize_text(StreamReader &stream) = 0;
    };
}


#endif //TENSORSTACK_MODULE_SERIALIZATION_H
