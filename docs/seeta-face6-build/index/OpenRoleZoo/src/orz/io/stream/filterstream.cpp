#include "orz/io/stream/filterstream.h"



namespace orz {

int64_t FilterInputStream::read(char *buf, int64_t len) {
    if(m_in != nullptr) {
        return m_in->read(buf, len);
    }
    return -1;
}


FilterInputStream::FilterInputStream(std::shared_ptr<InputStream> in):m_in(in) {
}


/////////////////////////////////////
int64_t FilterOutputStream::write(const char *buf, int64_t len) {
    if(m_out != nullptr) {
        return m_out->write(buf, len);
    }
    return -1;
}


FilterOutputStream::FilterOutputStream(std::shared_ptr<OutputStream> out):m_out(out) {
}





}
