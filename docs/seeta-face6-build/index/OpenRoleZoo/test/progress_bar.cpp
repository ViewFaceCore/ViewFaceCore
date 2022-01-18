//
// Created by lby on 2018/1/24.
//

#include <thread>
#include "orz/tools/progress_bar.h"
#include "orz/utils/log.h"

int main()
{
    orz::progress_bar bar(0, 10000);
    bar.start();
    for (int i = 0; i < 100000000000000; ++i) {
        if (bar.stat() == orz::progress_bar::STOPPED) break;
        bar.wait_show(1000, std::cout);

        std::chrono::milliseconds dura(1);
        std::this_thread::sleep_for(dura);

        bar.next();
    }
    bar.show(std::cout) << std::endl;

    ORZ_LOG(orz::INFO) << "Full takes " << orz::to_string(bar.used_time());

    return 0;
}
