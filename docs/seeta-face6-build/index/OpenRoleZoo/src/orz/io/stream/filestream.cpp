#include "orz/io/stream/filestream.h"
#include "orz/utils/log.h"
#include "orz/io/dir.h"

namespace orz {

int64_t FileInputStream::read(char *buf, int64_t len) {
    int nread = -1;
    if(m_in.is_open()) {
        m_in.read(buf, len);
        if(!m_in.bad()) {
            nread = m_in.gcount();
        }
    }

    if(nread > 0) {
       return nread;
    }else {
       if(m_in.eof()) {
           return 0;
       }else{
           return -1;
       }
    }
}


FileInputStream::FileInputStream(const std::string & file, bool istxt):m_file(file),
  m_in(file.c_str(), istxt ? std::ios_base::in : std::ios_base::binary) {
    if(!m_in.is_open()) {
        orz::Log(ERROR) << "open file:" << file  << " failed" << orz::crash;
    }
   
    std::string str = "/";//orz::FileSeparator();
    int nfind = m_file.find_last_of(str);
    if(nfind >= 0) {
        m_workpath = m_file.substr(0, nfind + 1); 
    }else {
        str = "\\";
        nfind = m_file.find_last_of(str);
        if(nfind >= 0) {
            m_workpath = m_file.substr(0, nfind + 1); 
        }
    }
 
}

FileInputStream::~FileInputStream() {
}


const char * FileInputStream::getworkpath(){
    return m_workpath.c_str(); 
}

/////////////////////////////////////
int64_t FileOutputStream::write(const char *buf, int64_t len) {
    int64_t nwrite = -1;
    if(m_out.is_open()) {
        m_out.write(buf, len);
        if(!m_out.bad()) {
            nwrite = len;
        }
    }
    return nwrite;

}



FileOutputStream::FileOutputStream(const std::string & file, bool istxt):m_file(file),
   m_out(file.c_str(), istxt ? std::ios_base::out : std::ios_base::binary) {
    if(!m_out.is_open()) {
        orz::Log(ERROR) << "open file:" << file  << " failed" << orz::crash;
    }
    
    std::string str = "/";//orz::FileSeparator();
    int nfind = m_file.find_last_of(str);
    if(nfind >= 0) {
        m_workpath = m_file.substr(0, nfind + 1); 
    }else {
        str = "\\";//orz::FileSeparator();
        nfind = m_file.find_last_of(str);
        if(nfind >= 0) {
            m_workpath = m_file.substr(0, nfind + 1); 
        }
    }
 
}


FileOutputStream::~FileOutputStream() {
}


const char * FileOutputStream::getworkpath(){
    return m_workpath.c_str(); 
}



}
