#pragma once

#include "SeetaLANLock.h"
#include "SeetaLockFunction.h"
#include <orz/utils/log.h>

static inline void VerifyLAN()
{
    SeetaLock_VerifyLANParallel verify;

    if (!SeetaLockSafe_call(&verify))
    {
        orz::Log(orz::FATAL) << "Lock module hijacked" << orz::crash;
    }

    if (verify.out.errcode != 0)
    {
        SeetaLock_ErrorMessage error_message(verify.out.errcode);
        SeetaLockSafe_call(&error_message);
        orz::Log(orz::FATAL) << "SeetaLANLock failed(" << verify.out.errcode << "): " << error_message.out.message << orz::crash;
    }
}
