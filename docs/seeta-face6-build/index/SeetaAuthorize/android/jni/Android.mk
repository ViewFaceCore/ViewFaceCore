LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE := orz-prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../OpenRoleZoo/android/obj/local/$(TARGET_ARCH_ABI)/libORZ_static.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../OpenRoleZoo/include/
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := SeetaAuthorize

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/../*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/*.cpp)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/hidden/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/lock/

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib -lm -llog

LOCAL_STATIC_LIBRARIES += orz-prebuilt

include $(BUILD_SHARED_LIBRARY)
