//
// Created by lby on 2018/1/11.
//

#ifndef ORZ_NET_HTTP_H
#define ORZ_NET_HTTP_H

#include <string>

namespace orz {
    namespace http {
        enum VERB {
            POST,
            GET
        };
    }

    namespace header {
        enum TYPE {
            JSON,   ///< "Content-Type:application/json; charset=utf-8";
            FORM,   ///< "Content-Type:application/x-www-form-urlencoded; charset=utf-8";
        };
    }

    class URL {
    public:
        using self  = URL;

        URL(const std::string &url);

        const std::string &url() const { return m_url; }

        const std::string &protocol() const { return m_protocol; }

        const std::string &host() const { return m_host; }

        int port() const { return m_port; }

        const std::string &target() const { return m_target; }

        bool valid() const { return m_valid; }

    private:
        std::string m_url;        // http://www.localhost.com:8090/index.html?lang=zh_CN
        std::string m_protocol;   // http
        std::string m_host;       // www.web.com
        int m_port;               // 8090
        std::string m_target;     // /index.html?lang=zh_CN
        bool m_valid;
    };

    std::string http_request(const URL &url, http::VERB verb, const std::string &data = "");
    std::string http_request(const URL &url, http::VERB verb, const std::string &data, header::TYPE header);
    std::string http_request(const URL &url, http::VERB verb, const std::string &data, const std::string &header);
}


#endif //ORZ_NET_HTTP_H
