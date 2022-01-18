//
// Created by xif on 18-1-18.
//

#ifndef ORZ_NET_HTTP_LINUX_H
#define ORZ_NET_HTTP_LINUX_H

#if !(ORZ_PLATFORM_CC_MSVC || ORZ_PLATFORM_CC_MINGW)

#include "orz/utils/log.h"
#include "orz/io/jug/binary.h"
#include "orz/mem/need.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>

#include <cstring>
#include <map>
#include <cstdlib>

#ifndef ORZ_WITH_OPENSSL

#else   // ORZ_WITH_OPENSSL

#include <cstdio>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <cstdlib>

#endif  // !ORZ_WITH_OPENSSL

namespace orz {

    class linux_ssl_static_init
    {
    public:
        linux_ssl_static_init() {
#ifndef ORZ_WITH_OPENSSL
#else   // !ORZ_WITH_OPENSSL
            SSLeay_add_ssl_algorithms();
#endif  // !ORZ_WITH_OPENSSL
        }
    };

    class linux_ssl_stream {
    public:
        explicit linux_ssl_stream(int fd, bool with_ssl = false)
                : m_fd(fd), m_with_ssl(with_ssl) {
#ifndef ORZ_WITH_OPENSSL
            if (with_ssl) {
                ORZ_LOG(ERROR) << "Can not open a stream without OpenSSL." << crash;
            } else {
                return;
            }
#else   // !ORZ_WITH_OPENSSL
            if (!with_ssl) return;

            // SSL_load_error_strings();
            static linux_ssl_static_init static_init;

            m_ssl_ctx = SSL_CTX_new(SSLv23_client_method());
            if (m_ssl_ctx == nullptr) {
                ERR_print_errors_fp(stdout);
                ORZ_LOG(ERROR) << "Can not new SSL_CTX." << crash;
            }
            m_ssl = SSL_new(m_ssl_ctx);
            if (m_ssl == nullptr) {
                SSL_CTX_free(m_ssl_ctx);
                ERR_print_errors_fp(stdout);
                ORZ_LOG(ERROR) << "Can not new SSL." << crash;
            }
            SSL_set_fd(m_ssl, m_fd);
            if (SSL_connect(m_ssl) < 0) {
                SSL_free(m_ssl);
                SSL_CTX_free(m_ssl_ctx);
                ORZ_LOG(ERROR) << "Can not connect SSL to " << fd << crash;
            }
#endif   // !ORZ_WITH_OPENSSL
            // verify
        }

        bool const verify() {
            if (!m_with_ssl) return true;
#ifndef ORZ_WITH_OPENSSL
            ORZ_LOG(ERROR) << "Can not verify a stream without OpenSSL." << crash;
            return false;
#else   // !ORZ_WITH_OPENSSL
            auto cipher = SSL_get_cipher(m_ssl);
            UNUSED(cipher);
            auto cert = SSL_get_peer_certificate(m_ssl);
            if (cert == nullptr) return false;
            need free_cert(X509_free, cert);
            auto subject = X509_NAME_oneline(X509_get_subject_name(cert), nullptr, 0);
            auto issuer = X509_NAME_oneline(X509_get_issuer_name(cert), nullptr, 0);
#if OPENSSL_VERSION_NUMBER < 0x10100000L
            need free_subject(CRYPTO_free, subject);
            need free_issuer(CRYPTO_free, issuer);
#else
            need free_subject(free, subject);
            need free_issuer(free, issuer);
#endif
            // TODO: really verify cert
            return true;
#endif   // !ORZ_WITH_OPENSSL
        }

        ssize_t read(void *data, size_t length) {
            if (m_with_ssl) {
#ifndef ORZ_WITH_OPENSSL
                ORZ_LOG(ERROR) << "Can not read a stream without OpenSSL." << crash;
                return -1;
#else   // !ORZ_WITH_OPENSSL
                return SSL_read(m_ssl, data, static_cast<int>(length));
#endif  // !ORZ_WITH_OPENSSL
            }
            return ::read(m_fd, data, length);
        }

        ssize_t write(const void *data, size_t length) {
            if (m_with_ssl) {
#ifndef ORZ_WITH_OPENSSL
                ORZ_LOG(ERROR) << "Can not write a stream without OpenSSL." << crash;
                return -1;
#else   // !ORZ_WITH_OPENSSL
                return SSL_write(m_ssl, data, static_cast<int>(length));
#endif  // !ORZ_WITH_OPENSSL
            }
            return ::write(m_fd, data, length);
        }

        ssize_t readline(void *data, size_t length) {
            size_t offset = 0;
            while (true) {
                ssize_t read_size = read(reinterpret_cast<char *>(data) + offset, 1);
                if (read_size <= 0) return read_size;
                ++offset;
                if (offset >= length) return length;
                if (reinterpret_cast<char *>(data)[offset - 1] == '\n') return offset;
            }
        }

        bool eof() const {
            char mark;
            auto flag = ::recv(m_fd, &mark, 1, MSG_PEEK);
            if (flag > 0) return false;
            if (flag < 0 && errno == EWOULDBLOCK) return false;
            return true;
        }

        ~linux_ssl_stream() {
#ifdef ORZ_WITH_OPENSSL
            if (m_with_ssl) {
                SSL_shutdown(m_ssl);
                SSL_free(m_ssl);
                SSL_CTX_free(m_ssl_ctx);
            }
#endif  // ORZ_WITH_OPENSSL
        }
    private:
        linux_ssl_stream(const linux_ssl_stream &other) = delete;
        const linux_ssl_stream &operator=(const linux_ssl_stream &other) = delete;

        int m_fd = 0;
        bool m_with_ssl = false;
#ifdef ORZ_WITH_OPENSSL
        SSL_CTX *m_ssl_ctx = nullptr;
        SSL *m_ssl = nullptr;
#endif  // ORZ_WITH_OPENSSL

    };

    using stream = linux_ssl_stream;

    static std::string tolower(const std::string &str) {
        std::string str_copy = str;
        for (auto &ch : str_copy) ch = char(std::tolower(ch));
        return std::move(str_copy);
    }

    static std::string trim(const std::string &str) {
        size_t left = 0;
        size_t right = str.size();
        while (right > left && str[left] == ' ') ++left;
        while (right > left && str[right - 1] == ' ') --right;
        return str.substr(left, right - left);
    }

    class linux_http_option {
    public:
        void add(const std::string &opt) {
            auto anchor = opt.find(':');
            if (anchor == std::string::npos) return;
            auto key = trim(opt.substr(0, anchor));
            auto value = trim(opt.substr(anchor + 1));
            add(key, value);
        }
        void add(const std::string &key, const std::string &value) {
            std::string lower_key = tolower(key);
            auto it = m_opts.find(lower_key);
            if (it == m_opts.end()) {
                std::vector<std::string> value_list(1, value);
                m_opts.insert(std::make_pair(lower_key, std::move(value_list)));
            } else {
                std::vector<std::string> &value_list = it->second;
                value_list.push_back(value);
            }
        }
        const std::vector<std::string> &get(const std::string &key) {
            std::string lower_key = tolower(key);
            auto it = m_opts.find(lower_key);
            static const std::vector<std::string> empty_list;
            if (it == m_opts.end()) {
                return empty_list;
            } else {
                return it->second;
            }
        }
        const std::string get_one(const std::string &key) {
            auto value_list = get(key);
            if (value_list.empty()) {
                return std::string();
            } else {
                return value_list.front();
            }
        }
    private:
        std::map<std::string, std::vector<std::string> > m_opts;
    };

    using option = linux_http_option;

    static int defulat_protocol_port(const std::string &protocal, int port) {
        if (port > 0) return port;
        std::string local_protocal = tolower(protocal);
        if (local_protocal == "http") return 80;
        if (local_protocal == "https") return 443;
        return 0;
    }

    static bool need_ssl(const std::string &protocal) {
        std::string local_protocal = tolower(protocal);
        if (local_protocal == "https") return true;
        return false;
    }

    static const char *verb_string(http::VERB verb) {
        switch (verb) {
            case http::GET:
                return "GET";
            case http::POST:
                return "POST";
            default:
                return "GET";
        }
    }

    static struct hostent *gethostbyname_local(const char *name, struct hostent &result_buf) {
        struct hostent *result = nullptr;
#if ORZ_PLATFORM_OS_IOS || ORZ_PLATFORM_OS_MAC
        result=gethostbyname(name);
#else
        char buf[1024];
        size_t buflen = sizeof(buf);
        int err;
        if (gethostbyname_r(name, &result_buf, buf, buflen, &result, &err) != 0 || result == nullptr) {
            return nullptr;
        }
#endif
        return result;
    }

    static bool getline(stream &s, std::string &line) {
        char data[1024];
        ssize_t len = 0;
        line.clear();
        while (true) {
            len = s.readline(data, sizeof(data));
            if (len < 0) break;
            if (len == 0) {
                if (s.eof()) break;
                else continue;
            }
            if (data[len - 1] == '\n') {
                 int tail = 1;
                 if (len >= 2 && data[len - 2] == '\r') tail++;
                 line.insert(line.end(), data, data + len - tail);
                // line.insert(line.end(), data, data + len);
                break;
            } else {
                line.insert(line.end(), data, data + len);
            }
        }
        if (line.empty() && len < 0) return false;
        return true;
    }

    static binary read_closed_stream(stream &s) {
        binary data;
        char local_buffer[1024];
        ssize_t local_length = 0;
        while (true) {
            local_length = s.read(local_buffer, sizeof(local_buffer));
            if (local_length < 0) break;
            if (local_length == 0) {
                if (s.eof()) break;
                else continue;
            }
            data.write(local_buffer, local_length);
        }
        return data;
    }

    static binary read_content_data(stream &s, size_t length) {
        binary data;
        data.resize(length);
        while (data.get_pos() < data.size()) {
            auto remain_size = data.size() - data.get_pos();
            auto read_size = s.read(data.data<char>() + data.get_pos(), remain_size);
            if (read_size == 0 && s.eof()) break;
            data.shift(read_size);
        }
        return data;
    }

    static binary read_chunked_data(stream &s) {
        binary data;
        binary buff;
        std::string chunk_header;
        while (getline(s, chunk_header)) {
            if (chunk_header.empty()) break;
            auto chunk_length = std::strtol(chunk_header.c_str(), nullptr, 16);
            buff = read_content_data(s, chunk_length);
            data.write(buff.data(), buff.size());
            getline(s, chunk_header);
            if (chunk_length == 0) break;
        }
        return data;
    }

    std::string http_request_core(const URL &url, http::VERB verb, const std::string &data, const std::string &header) {
        std::string report;

        struct hostent host_buf = {nullptr};
        struct hostent *host = gethostbyname_local(url.host().c_str(), host_buf);
        if (host == nullptr)
        {
            ORZ_LOG(ERROR) << "gethostbyname error" << crash;
        }
        in_addr host_addr = *reinterpret_cast<in_addr *>(host->h_addr);
        std::string ip = inet_ntoa(host_addr);

        std::string target = url.target();
        if (target.empty()) target = "/";

        std::ostringstream content_buffer;
        content_buffer << Concat(verb_string(verb), " ", target, " HTTP/1.1\r\n");
        content_buffer << Concat("HOST: ", ip, "\r\n");
        content_buffer << Concat("User-Agent: ", "Microsoft Internet Explorer", "\r\n");
        // content_buffer << Concat("Connection: ", "keep-alive", "\r\n");
        content_buffer << Concat(header, "\r\n");
        content_buffer << Concat("Content-Length: ", data.size(), "\r\n");
        content_buffer << "\r\n";
        content_buffer << data;
        std::string content = content_buffer.str();
        // content.insert(content.end(), data.begin(), data.end());

        int sockfd = socket(host->h_addrtype, SOCK_STREAM, 0);
        if (sockfd < 0) ORZ_LOG(ERROR) << "socket error" << crash;
        need close_sockfd(close, sockfd);


        struct sockaddr_in servaddr = {0};
        std::memset(&servaddr, 0, sizeof(servaddr));

        int port = defulat_protocol_port(url.protocol(), url.port());
        servaddr.sin_family = static_cast<sa_family_t >(host->h_addrtype);
        servaddr.sin_port = htons(port);
        servaddr.sin_addr = host_addr;

        if (connect(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0)
            ORZ_LOG(ERROR) << "connect error" << crash;

        bool with_ssl = need_ssl(url.protocol());
        stream socket_stream(sockfd, with_ssl);

        if (with_ssl && !socket_stream.verify()) {
            ORZ_LOG(ERROR) << "verify error" << crash;
        }

        socket_stream.write(content.data(), content.size());

        std::string line;
        option  http_option;

        while (getline(socket_stream, line)) {
            if (line.empty()) break;
            http_option.add(line.substr(0, line.size()));
        }

        std::string data_length = http_option.get_one("Content-Length");
        std::string data_encoding = http_option.get_one("Transfer-Encoding");

        binary buffer;

        if (!data_length.empty()) {
            auto idata_length = std::strtol(data_length.c_str(), nullptr, 10);
            buffer = read_content_data(socket_stream, idata_length);
        } else if (!data_encoding.empty() && data_encoding == "chunked") {
            buffer = read_chunked_data(socket_stream);
        } else {
            if (tolower(http_option.get_one("Connection")) != "close")
                ORZ_LOG(ERROR) << "Can not recognize content format";
            buffer = read_closed_stream(socket_stream);
        }

        report = std::string(buffer.data<char>(), buffer.size());

        return report;
    }
}

#else   // !(ORZ_PLATFORM_CC_MSVC || ORZ_PLATFORM_CC_MINGW)

#error Can not compile with msvc or mingw

#endif  // !(ORZ_PLATFORM_CC_MSVC || ORZ_PLATFORM_CC_MINGW)

#endif //ORZ_NET_HTTP_LINUX_H
