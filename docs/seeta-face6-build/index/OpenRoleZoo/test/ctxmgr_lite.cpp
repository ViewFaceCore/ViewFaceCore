//
// Created by kier on 2018/11/11.
//

#include "orz/tools/ctxmgr_lite_support.h"
#include "orz/utils/log.h"

ORZ_LITE_CONTEXT(int)

void print_context_int() {
    ORZ_LOG(orz::INFO) << orz::ctx::lite::ref<int>();
}

int main() {
    int a = 12;
    orz::ctx::lite::bind<int> _bind_int(a);
    print_context_int();
}