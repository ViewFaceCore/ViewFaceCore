//
// Created by xif on 18-2-9.
//


#include <orz/io/dir.h>
#include <orz/io/walker.h>
#include <orz/utils/log.h>

int main(int argc, char *argv[]) {
    auto result = orz::copy("1.jpg", "3.jpg");
    ORZ_LOG(orz::INFO) << "Copy 1.jpg to 3.jpg, status: " << std::boolalpha << result;

    ORZ_LOG(orz::INFO) << "Current work dir: \"" << orz::getcwd() << "\"";
    ORZ_LOG(orz::INFO) << "Self module name: \"" << orz::getself() << "\"";
    ORZ_LOG(orz::INFO) << "Current exe name: \"" << orz::getexed() << "\"";

    ORZ_LOG(orz::INFO) << "Change work dir to ..";

    orz::cd("..");

    ORZ_LOG(orz::INFO) << "Current work dir: \"" << orz::getcwd() << "\"";
    ORZ_LOG(orz::INFO) << "Self module name: \"" << orz::getself() << "\"";
    ORZ_LOG(orz::INFO) << "Current exe name: \"" << orz::getexed() << "\"";



    return 0;
}