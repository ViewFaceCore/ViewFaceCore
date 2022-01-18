//
// Created by lby on 2018/1/11.
//

#include "orz/net/http.h"
#include "orz/utils/platform.h"

#if ORZ_PLATFORM_CC_MSVC || ORZ_PLATFORM_CC_MINGW

#include "http_win.h"

#else
#include "http_linux.h"
#endif

namespace orz {
    URL::URL(const std::string &url)
            : m_url(url), m_valid(false) {
        auto anchor = m_url.find(':', 0);
        if (anchor != std::string::npos) {
            if (anchor + 2 < m_url.length() && m_url[anchor + 1] == '/' && m_url[anchor + 2] == '/') {
                m_protocol = m_url.substr(0, anchor);
                anchor += 3;
            } else {
                anchor = 0;
            }
        } else {
            anchor = 0;
        }
        auto target_anchor = m_url.find('/', anchor);
        auto port_anchor = m_url.find(':', anchor);
        if (port_anchor <= target_anchor) {
            m_host = m_url.substr(anchor, port_anchor - anchor);
            std::string port_string;
            if (target_anchor == std::string::npos) port_string = m_url.substr(port_anchor + 1);
            else port_string = m_url.substr(port_anchor + 1, target_anchor - port_anchor);
            m_port = std::atoi(port_string.c_str());
        } else {
            m_host = m_url.substr(anchor, target_anchor - anchor);
            m_port = 0;
        }
        if (target_anchor != std::string::npos) {
            m_target = m_url.substr(target_anchor);
        }

        m_valid = true;
    }

    std::string http_request(const URL &url, http::VERB verb, const std::string &data, const std::string &header) {
        try {
            return http_request_core(url, verb, data, header);
        } catch (const Exception &) {
            return std::string();
        }
    }

    std::string http_request(const URL &url, http::VERB verb, const std::string &data, header::TYPE header) {
        std::string header_str;
        switch (header) {
            case header::JSON:
                header_str = "Content-Type:application/json; charset=utf-8";
                break;
            case header::FORM:
                header_str = "Content-Type:application/x-www-form-urlencoded; charset=utf-8";
                break;
        }
        return http_request(url, verb, data, header_str);
    }

    std::string http_request(const URL &url, http::VERB verb, const std::string &data) {
        return http_request(url, verb, data, header::FORM);
    }
}