#include "orz/io/stream/bufferedstream.h"
#include "orz/utils/log.h"


namespace orz {

int64_t BufferedInputStream::read(char *buf, int64_t len) {
    if(m_in == nullptr) {
        return -1;
    }

    if(len < 0) {
        return -1;
    }else if(len == 0) {
        return 0;
    }

    if(len <= m_datalen - m_pos) {
        memcpy(buf, m_buf.get() + m_pos, len);
        m_pos += len;
        return len;
    }

    if(m_datalen - m_pos > 0) {    
        memcpy(buf, m_buf.get() + m_pos, m_datalen - m_pos);
    }
    int64_t num = m_datalen - m_pos;
    m_pos = 0;
    m_datalen = 0;
    int64_t n = 0;
  
    for(;;) {
        n = m_in->read(m_buf.get(), m_len);
        if(n > 0) {
            if(len <= num + n) {
                memcpy(buf + num, m_buf.get(), len - num);
                m_pos = len - num;
                m_datalen = n;
                num += m_pos;
                break;
            }else {
                memcpy(buf + num, m_buf.get(), n);
                num += n;
                continue; 
            }
        }else {
            break;
        }

    }

    if(num > 0)
    {
        return num;
    }else {
        return n;
    }
}


BufferedInputStream::BufferedInputStream(std::shared_ptr<InputStream> in, int64_t size):FilterInputStream(in) {
    m_len = size;
    m_pos = 0;
    m_datalen = 0;
    m_buf = std::shared_ptr<char>(new char[m_len], std::default_delete<char[]>());  
}

/////////////////////////////////////
int64_t BufferedOutputStream::write(const char *buf, int64_t len) {
    if(m_out == nullptr) {
        return -1;
    }

    if(m_len - m_pos >= len) {
        memcpy(m_buf.get() + m_pos, buf, len);
        m_pos += len;
        return len;
    }

    if(m_len - m_pos > 0) { 
        memcpy(m_buf.get() + m_pos, buf, m_len - m_pos);
    }
    int64_t num = m_len - m_pos; 
    m_pos += num;
    flush();

    for(;;) {
        if(len - num > m_len) {
            memcpy(m_buf.get(), buf + num, m_len);
            num += m_len;
            flush();
            continue;
        }else {
            memcpy(m_buf.get(), buf + num, len - num);
            m_pos = len - num;
            num += m_pos;
            break;
        }
    }
    return num;
}


void BufferedOutputStream::flush() {
    if(m_out == nullptr) {
        return;
    }

    if(m_pos <= 0)
        return;

    int64_t n = m_out->write(m_buf.get(), m_pos);
    if( n != m_pos) {
        orz::Log(orz::ERROR) << "BufferedOutputStream flush failed" << orz::crash;
    }
    m_pos = 0;
}



BufferedOutputStream::BufferedOutputStream(std::shared_ptr<OutputStream> out, int64_t size):FilterOutputStream(out) {
    m_len = size;
    m_pos = 0;
    m_buf = std::shared_ptr<char>(new char[m_len], std::default_delete<char[]>());  
}

BufferedOutputStream::~BufferedOutputStream() {
    flush();
}




}
