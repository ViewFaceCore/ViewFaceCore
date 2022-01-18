//
// Created by kier on 2018/11/7.
//

#include <module/io/fstream.h>
#include "aes_fstream.h"
#include <string.h>
#include <utils/assert.h>
#include <utils/log.h>

namespace ts {
    bool AESFileStreamReader::is_open() const {
        return m_stream.is_open();
    }

    size_t AESFileStreamReader::read(void *buffer, size_t size) {
        if(m_master_datalen - m_master_offset >= size) {
            memcpy(buffer, m_master + m_master_offset, size);
            m_master_offset += int(size);
            return size; 
        }

        bool iseof = false;
        int buffer_offset= 0;
        if(m_master_datalen - m_master_offset > 0) {
            memcpy(buffer, m_master + m_master_offset, m_master_datalen - m_master_offset);
            buffer_offset = m_master_datalen - m_master_offset;
            m_master_offset = 0;
            m_master_datalen = 0;
        }

        if(m_stream.eof()) {
            if(buffer_offset > 0) {
                return size_t(buffer_offset);
            }else
            {
                TS_LOG_ERROR << "mode file is eof!" << eject;
                return 0;
            }
        }
 
        while(buffer_offset < size) {    
            if(m_second_datalen == 0 ) {
                m_stream.read(reinterpret_cast<char *>(m_second), AES_BLOCKLEN);
                m_second_datalen = int(m_stream.gcount());
                if(m_second_datalen == 0 ) {
                    if(m_stream.eof()) {
                        break;
                    }else {
                        TS_LOG_ERROR << "mode file read format is error!" << eject;
                        return 0;
                    }
                }else {
                    if(m_second_datalen != AES_BLOCKLEN) {
                        TS_LOG_ERROR << "mode file read format is error!" << eject;
                        return 0;
                    }
                    AES_ECB_decrypt(&m_ctx, m_second);  
                }
            }

            memcpy(m_master, m_second, AES_BLOCKLEN);
            m_master_datalen = AES_BLOCKLEN;
            m_master_offset = 0;

            m_second_datalen = 0; 
            m_stream.read(reinterpret_cast<char *>(m_second), AES_BLOCKLEN);
            m_second_datalen = int(m_stream.gcount());
            if(m_second_datalen == 0 ) {
                if(m_stream.eof()) {
                    iseof = true;
                }else {
                    TS_LOG_ERROR << "mode file read format is error!" << eject;
                    return 0;
                }
            }else {
                if(m_second_datalen != AES_BLOCKLEN) {
                    TS_LOG_ERROR << "mode file read format is error!" << eject;
                    return 0;
                }
                AES_ECB_decrypt(&m_ctx, m_second);  
            }
 
            if(iseof) {
                m_master_datalen -= (int)(m_master[AES_BLOCKLEN - 1]); 
            } 

            
            if(buffer_offset + m_master_datalen >= size) {
                memcpy(reinterpret_cast<char*>(buffer) + buffer_offset, m_master, size - buffer_offset);
                m_master_offset = int(size) - buffer_offset;
                buffer_offset += m_master_offset;
                break;
            }else {
                memcpy(reinterpret_cast<char*>(buffer) + buffer_offset, m_master, m_master_datalen);
                buffer_offset += m_master_datalen;
                m_master_offset = 0;
                m_master_datalen = 0;
            } 

            if(iseof){
                break;
            }
        }
        
        return size_t(buffer_offset);
    }

    AESFileStreamReader::AESFileStreamReader(const std::string &path, const std::string &key)
            : m_stream(path, std::ios::binary) {
         m_master_offset = 0;
         m_master_datalen = 0;
         m_second_datalen = 0;

        if (key.length() > AES_KEYLEN) {
            TS_LOG_ERROR << "Using key over " << AES_KEYLEN << " will be ignored.";
        }

         AES_init_ctx(&m_ctx, (uint8_t*)key.c_str(), uint32_t(key.length()));
    }

    void AESFileStreamReader::close() {
        m_stream.close();
    }

    AESFileStreamReader::~AESFileStreamReader() {
        close();
    }


    ////////////////////////////////////////////////////////////
    bool AESFileStreamWriter::is_open() const {
        return m_stream.is_open();
    }

    size_t AESFileStreamWriter::write(const void *buffer, size_t size) {

        size_t nwrite = 0;
        while(nwrite < size ) {
            if(AES_BLOCKLEN - m_master_datalen  >= size - nwrite) {
                memcpy(m_master + m_master_datalen, reinterpret_cast<const char *>(buffer) + nwrite, size - nwrite);
                m_master_datalen += int(size - nwrite);
                nwrite = size;
                return nwrite; 
            }else {
                memcpy(m_master + m_master_datalen, reinterpret_cast<const char *>(buffer) + nwrite, AES_BLOCKLEN - m_master_datalen);
                AES_ECB_encrypt(&m_ctx, m_master);  
                m_stream.write(reinterpret_cast<const char *>(m_master), AES_BLOCKLEN);
                nwrite += AES_BLOCKLEN - m_master_datalen; 
                m_master_datalen = 0;
                if(m_stream.bad()) {
                    return 0;
                }
            }
        }
        return nwrite;
    }

    AESFileStreamWriter::AESFileStreamWriter(const std::string &path, const std::string &key)
            : m_stream(path, std::ios::binary) {
         m_master_datalen = 0;

         if (key.length() > AES_KEYLEN) {
             TS_LOG_ERROR << "Using key over " << AES_KEYLEN << " will be ignored.";
         }

         AES_init_ctx(&m_ctx, (uint8_t*)key.c_str(), uint32_t(key.length()));
    }

    AESFileStreamWriter::~AESFileStreamWriter() {
         close();
    }

    void AESFileStreamWriter::close() {
        if(!m_stream.is_open()) {
            return;
        }

        if(m_master_datalen == AES_BLOCKLEN) {
            AES_ECB_encrypt(&m_ctx, m_master);  
            m_stream.write(reinterpret_cast<const char *>(m_master), AES_BLOCKLEN);
            m_master_datalen = 0;
            if(m_stream.bad()) {
                TS_LOG_ERROR << "write mode file failed!" << eject;
                return;
            }
        } 
        uint8_t npadding = (uint8_t)(AES_BLOCKLEN - m_master_datalen);
        uint8_t buf[AES_BLOCKLEN];
        memset(buf, npadding, AES_BLOCKLEN);
        memcpy(buf, m_master, m_master_datalen);
        AES_ECB_encrypt(&m_ctx, buf);  
        
        m_stream.write(reinterpret_cast<const char *>(buf), AES_BLOCKLEN);
        m_master_datalen = 0;
        m_stream.close();
    }

}
