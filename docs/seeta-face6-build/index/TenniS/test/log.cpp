//
// Created by kier on 2018/11/6.
//

#include <utils/log.h>
#include <utils/assert.h>

int main() {
    TS_LOG(ts::LOG_INFO) << "Show log";
    TS_TIME(ts::LOG_INFO) << "Show time";
    TS_LOG_TIME(ts::LOG_INFO) << "Show log time";

    try {
        TS_CHECK(0 > 1) << "0 greater than 1" << ts::eject;
    } catch (const ts::Exception &e) {

    }

    try {
        TS_CHECK_EQ(0, 1) << ts::fatal;
    } catch (const ts::Exception &e) {

    }

}