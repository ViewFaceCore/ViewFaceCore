//
// Created by kier on 2018/11/7.
//

#include <module/io/fstream.h>
#include <utils/log.h>

int main() {
    int a[] = {1, 2, 3};

    ts::FileStreamWriter out("test.bin");
    ts::binio::write(out, a[0]);
    out.close();

    int temp = 0;
    ts::FileStreamReader in("test.bin");
    ts::binio::read(in, temp);

    TS_LOG(ts::LOG_INFO) << temp;
}

