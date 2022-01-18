cmake_minimum_required(VERSION 2.8)
set(FaceTracking_NAME "SeetaFaceTracking" CACHE STRING "The FaceTracking library name")
set(FaceTracking_VERSION_MAJOR "6")
set(FaceTracking_VERSION_MINOR "0")
set(FaceTracking_VERSION_SINOR "0")
set(FaceTracking_VERSION "${FaceTracking_VERSION_MAJOR}${FaceTracking_VERSION_MINOR}${FaceTracking_VERSION_SINOR}")
project(${FaceTracking_NAME}${FaceTracking_VERSION})
#set(CMAKE_VERBOSE_MAKEFILE on)

set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# option for platform
set(PLATFORM "x64" CACHE STRING "x86 or x64")
set(CONFIGURATION "Debug" CACHE STRING "Debug or Release")
set(CONFIGURATION ${CMAKE_BUILD_TYPE})

# option for shared or static
set(TARGET "SHARED" CACHE STRING "SHARED or STATIC" FORCE)
# option for SeetaAuthorize
option(SEETA_AUTHORIZE "Using seeta authorize" ON)
# option for build android
option(BUILD_ANDROID "Buid android" OFF)
message(STATUS "Build android: " ${BUILD_ANDROID})

# gether moduls
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/../build/cmake)
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# install dir
set(CMAKE_INSTALL_PREFIX ${SOLUTION_DIR}/../build CACHE STRING "set install prefix" FORCE)
set(SEETA_INSTALL_PATH "" CACHE STRING "seeta install path")
if(SEETA_INSTALL_PATH STREQUAL "")
else()
	set(CMAKE_INSTALL_PREFIX ${SEETA_INSTALL_PATH} CACHE STRING "set install prefix" FORCE)
endif()
message(STATUS "cmake install prefix: ${CMAKE_INSTALL_PREFIX}")

# set base flags
set(CMAKE_CFLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -w -std=c++11")

# env
include(LOCAL_ENV)
include(INC_ORZ)

if ("${PLATFORM}" STREQUAL "x86")
    message(STATUS "Platform: " ${PLATFORM})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
elseif ("${PLATFORM}" STREQUAL "x64")
    message(STATUS "Platform: " ${PLATFORM})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
else()
	message(STATUS "Platform: " ${PLATFORM})
endif()

if ("${CONFIGURATION}" STREQUAL "Debug")
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_CX_FLAGS "${CMAKE_CX_FLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
else()
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_CX_FLAGS "${CMAKE_CX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

add_subdirectory(${SOLUTION_DIR}/FaceTracking)
if(NOT BUILD_ANDROID)
	#add_subdirectory(${SOLUTION_DIR}/example)
endif()
