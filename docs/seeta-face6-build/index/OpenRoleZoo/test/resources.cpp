//
// Created by kier on 2018/9/25.
//

#include <orz/tools/resources.h>
#include <iostream>

int main() {
    orz::resources::compiler compiler;
    bool succeed = compiler.compile("model", "orz_resources.h", "orz_resources.c");
    if (!succeed) {
        std::cerr << compiler.last_error_message() << std::endl;
    }

    return 0;
}


