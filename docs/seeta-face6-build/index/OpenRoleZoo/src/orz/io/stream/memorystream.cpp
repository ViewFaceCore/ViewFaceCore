#include "orz/io/stream/memorystream.h"

#include <string.h>
#include <limits>
#include "orz/utils/log.h"

namespace orz {

int64_t MemoryInputStream::read(char *buf, int64_t len) {
    if(m_buf != nullptr) {
        if(m_pos >= m_len) {
            return 0; 
        }

        int64_t avail = m_len - m_pos;
        if(len > avail) {
           len = avail;
        } 

        if(len <= 0) {
            return 0;
        }
       
        memcpy(buf, m_buf.get() + m_pos, len);
        m_pos += len;
        return len;
    }
    return -1;
}



MemoryInputStream::MemoryInputStream(const char *buf, int64_t len) {
    m_buf = std::shared_ptr<char> (new char[len], std::default_delete<char []>());
    memcpy(m_buf.get(), buf, len);
    m_len = len;
    m_pos = 0;
}


bool MemoryInputStream::setpos(int64_t pos) {
    if(m_buf == nullptr) {
        return false;
    }

    if(pos > m_len) {
        return false;
    }
    m_pos = pos;
    return true;
}
/////////////////////////////////////
void MemoryOutputStream:: grow(int64_t mincapacity) {
    const int64_t max_size = std::numeric_limits<int64_t>::max();
    int64_t nold = m_len;
    int64_t ncur = 0;
    if(nold > (max_size >> 1)) {
       ncur = max_size;
    }else {
       ncur = nold << 1;
    }

    if(ncur - mincapacity < 0) {
        ncur = mincapacity;
    }

    char *ptr = new char[ncur];
    memcpy(ptr, m_buf.get(), m_pos);
    m_buf.reset(ptr, std::default_delete<char []>());
    m_len = ncur;
}

int64_t MemoryOutputStream::write(const char *buf, int64_t len) {
    if(m_buf != nullptr) {
        if(m_pos > std::numeric_limits<int64_t>::max() - len) {
            return -1;
        }
        if(m_pos + len > m_len) {
            grow(m_pos + len); 
        }

        memcpy(m_buf.get() + m_pos, buf, len);
        m_pos += len;
        return len;
    }
    return -1;
    
}


std::string MemoryOutputStream::getdata() {
    if(m_buf == nullptr) {
        return "";
    }
    return std::string(m_buf.get(), m_pos);
}

bool MemoryOutputStream::setpos(int64_t pos) {
    if(m_buf == nullptr) {
        return false;
    }

    if(pos > m_len) {
        return false;
    }
    m_pos = pos;
    return true;
}


MemoryOutputStream::MemoryOutputStream(int64_t size) {
    m_buf = std::shared_ptr<char> (new char[size], std::default_delete<char []>());
    m_len = size;
    m_pos = 0;
}





}
