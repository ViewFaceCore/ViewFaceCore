#include "orz/io/stream/modelfilestream.h"


#include "orz/utils/log.h"
#include "orz/io/stream/filestream.h"
#include "orz/io/stream/memorystream.h"
#include "orz/codec/json.h"
#include "orz/io/i.h"
#include <fstream>

namespace orz {

int64_t ModelFileInputStream::read(char *buf, int64_t len) {
    return m_in->read(buf, len);
}


ModelFileInputStream::ModelFileInputStream(const std::string & file) {
     unsigned int mask = 0;
     int64_t nreads = 0;
     m_size = 0;
     m_istxt = false;
     m_file = file;

     {
         std::ifstream reader(file.c_str(), std::ios_base::binary);
         if(!reader.is_open()){
             orz::Log(orz::ERROR) << "open the model file:" << file << " failed!" << orz::crash;
         }
         reader.read((char *)&mask, 4);
         if(reader.bad()) {
             orz::Log(orz::ERROR) << "read the model file:" << file << " failed!" << orz::crash;
         }
         reader.seekg(0, std::ios::end);
         m_size = reader.tellg();
             
         reader.close(); 
     } 

     if(mask == 0x19910929) {
         m_in = std::shared_ptr<InputStream>(new FileInputStream(file));
     }else {
         m_istxt = true;
         m_in = std::shared_ptr<InputStream>(new FileInputStream(file, true));
     } 
 
}

ModelFileInputStream::~ModelFileInputStream() {
}

orz::jug ModelFileInputStream::read_jug() {
    std::shared_ptr<char> buf = std::shared_ptr<char>(new char[m_size], std::default_delete<char []>());
    int64_t nreads = read(buf.get(), m_size);
    if(nreads <= 0) {
        orz::Log(orz::ERROR) << "read model file: " << m_file << " failed!" << orz::crash;
        return orz::jug(); 
    }
    if(m_istxt) {
        std::string str(buf.get(), nreads);
        std::string strpath = m_in->getworkpath();
        return json2jug(str, strpath);
    }else {
        orz::imemorystream in(buf.get() + 4, nreads - 4);
        auto model  = orz::jug_read(in);
        return model;
    }
}


orz::jug ModelFileInputStream::read_jug(std::shared_ptr<InputStream> in) {
    unsigned int mask = 0;
    int64_t nreads = 0;
    bool istxt = false;

    nreads = in->read((char *)&mask, 4);
    if(nreads != 4) {
        orz::Log(orz::ERROR) << "InputStream format is error!"<< orz::crash;
    }

    std::shared_ptr<InputStream> ifile;
    if(mask == 0x19910929) {
        ifile = in;
    }else if(mask == 0x61746573) {
        orz::Log(orz::ERROR) << "do not supported file type!"<< orz::crash;
    }else {
        istxt = true;
        ifile = in;
    }

    char buf[10240];
    int len = sizeof(buf);

    orz::MemoryOutputStream omemory(10240000);

    if(istxt) {
        omemory.write((const char *)&mask, 4);
    }

    while(1) {
        nreads = ifile->read(buf, len);
        if(nreads > 0) {
            if(omemory.write(buf, nreads) != nreads) {
                orz::Log(orz::ERROR) << "write memoryoutputstream failed!" << orz::crash;
            }
        }else {
            break;
        }
    }

    std::string data = omemory.getdata();
    if(istxt) {
        std::string strpath = ifile->getworkpath();
        return json2jug(data, strpath);
    }else {
        orz::imemorystream in(data.data(), data.length());
        auto model  = orz::jug_read(in);
        return model;
    }
    return orz::jug();
}



/////////////////////////////////////
int64_t ModelFileOutputStream::write(const char *buf, int64_t len) {
    int64_t nwrite = -1;
    nwrite = m_out->write(buf, len);
    return nwrite;
}


ModelFileOutputStream::ModelFileOutputStream(const std::string & file, bool istxt) {
    m_istxt = istxt;

    if(m_istxt) {
        m_out = std::shared_ptr<OutputStream>(new FileOutputStream(file, true));
    }else {
        m_out = std::shared_ptr<OutputStream>(new FileOutputStream(file));
    }    
}


ModelFileOutputStream::~ModelFileOutputStream() {
}





}
