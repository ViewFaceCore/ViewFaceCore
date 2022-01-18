//
// Created by lby on 2018/1/10.
//

#include <orz/codec/json.h>
#include <orz/utils/log.h>

int main(int argc, char *argv[]) {
    std::string json_string =
            R"({
    "name": "json2sta",
    "date": "@date",
    "stride": 4,
    "scale": 1.414,
    "threshold": [
        0.7,
        0.7,
        0.85
    ],
    "transform": {
        "type": "sigmoid",
        "param": [
            1,
            2.3
        ]
    },
    "end2end" : false,
    "data" : "@file@1.jpg"
})";
    auto value = orz::json2jug(json_string);
    ORZ_LOG(orz::INFO) << "Original json: " << json_string;
    ORZ_LOG(orz::INFO) << "Converted jug: " << value;
    return 0;
}