//
// Created by kier on 2018/11/11.
//

#include <module/io/fstream.h>
#include <core/tensor.h>
#include <core/tensor_builder.h>
#include <utils/log.h>
#include <global/setup.h>
#include <module/module.h>

int main() {
    ts::setup();

    ts::Tensor str = ts::tensor::from("ABC");
//    ts::Bubble bubble("str:op", "str:name", 3);
    ts::Bubble bubble("str:op", "str:name", 1);
    bubble.set("str:param", str);

    ts::FileStreamWriter out("test.txt");
    str.serialize(out);
    bubble.serialize(out);
    out.close();

    ts::Tensor temp;
    ts::Bubble b;
    ts::FileStreamReader in("test.txt");
    temp.externalize(in);
    b.externalize(in);

    TS_LOG(ts::LOG_INFO) << ts::tensor::to_string(temp);
    TS_LOG(ts::LOG_INFO) << b;
    TS_LOG(ts::LOG_INFO) << ts::tensor::to_string(b.get("str:param"));
}

