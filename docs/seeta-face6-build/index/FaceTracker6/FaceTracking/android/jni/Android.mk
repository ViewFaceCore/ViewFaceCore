LOCAL_PATH := $(call my-dir)/../..

include $(CLEAR_VARS)
LOCAL_MODULE := fd-prebuilt
LOCAL_SRC_FILES := ${LOCAL_PATH}/../../FaceBoxes/FaceDetector/android/libs/$(TARGET_ARCH_ABI)/libSeetaFaceDetector600.so
LOCAL_EXPORT_C_INCLUDES := ${LOCAL_PATH}/../../FaceBoxes/FaceDetector/include/
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := openrolezoo-prebuilt
LOCAL_SRC_FILES := ${LOCAL_PATH}/../../OpenRoleZoo/android/obj/local/$(TARGET_ARCH_ABI)/libORZ_static.a
LOCAL_EXPORT_C_INCLUDES += ${LOCAL_PATH}/../../OpenRoleZoo/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := SeetaFaceTracking600

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/seeta/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/src/seeta/*.cpp)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/include

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib

LOCAL_LDLIBS += -llog -lz

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
    LOCAL_CFLAGS += -mfloat-abi=softfp
endif

LOCAL_CFLAGS += -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize  -ffast-math

LOCAL_SHARED_LIBRARIES += fd-prebuilt

LOCAL_STATIC_LIBRARIES += openrolezoo-prebuilt

include $(BUILD_SHARED_LIBRARY)
