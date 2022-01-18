LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)

LOCAL_MODULE := tennis

LOCAL_CFLAGS += -DTS_ON_ARM=ON
LOCAL_CFLAGS += -DTS_USE_OPENMP=ON
LOCAL_CFLAGS += -DTS_USE_NEON=ON
LOCAL_CFLAGS += -DTS_USE_SIMD=ON

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
	LOCAL_CFLAGS += -DTS_ON_ARMV7=ON
endif

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/../src/api/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/base/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/dragon/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/mxnet/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/onnx/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/tf/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/torch/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/backend/zoo/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/board/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/compiler/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/compiler/option/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/core/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/core/sync/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/encryption/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/engine/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/frontend/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/global/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/common/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/common/third/dragon/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/cpu/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/cpu/arm/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/cpu/caffe/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/cpu/dcn/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/cpu/dragon/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/kernels/cpu/quantized/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/memory/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/memory/orz/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/module/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/module/io/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/runtime/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/runtime/inferer/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/runtime/inside/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/runtime/instruction/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/utils/*.cpp)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../src/

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib

LOCAL_LDLIBS += -llog -fopenmp

LOCAL_CFLAGS += -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize  -ffast-math -fopenmp

LOCAL_SHORT_COMMANDS := true
	
include $(BUILD_SHARED_LIBRARY)
