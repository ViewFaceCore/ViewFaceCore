cmake_minimum_required(VERSION 2.8)
project(SeetaQualityAssessor300)

set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# gether moduls
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/../build/cmake)
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/cmake)

# install dir
set(CMAKE_INSTALL_PREFIX ${SOLUTION_DIR}/../build CACHE STRING "set install prefix" FORCE)

# set base flags
set(CMAKE_CFLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -Wall -std=c++11")
set(CMAKE_CFLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# option for platform
set(PLATFORM "x64" CACHE STRING "auto, x86 or x64")
set(CONFIGURATION "Debug" CACHE STRING "Debug or Release")
set(CONFIGURATION "${CMAKE_BUILD_TYPE}")

# option for SeetaAuthorize
option(SEETA_AUTHORIZE "Using seeta authorize" OFF)
option(SEETA_MODEL_ENCRYPT "Using model encrypt" ON)

# env
include(LOCAL_ENV)
include(INC_ORZ)

if(SEETA_AUTHORIZE)
	add_definitions(-DSEETA_AUTO_ENCRYPT)
	set(SEETA_AUTO_ENCRYPT ON)
	message(STATUS "Seeta Authorize: ON")
else()
	message(STATUS "Seeta Authorize: OFF")
endif()

if(SEETA_MODEL_ENCRYPT)
	add_definitions(-DSEETA_MODEL_ENCRYPT)
	set(SEETA_MODEL_ENCRYPT ON)
	message(STATUS "Seeta Model Encrypt: ON")
else()
	message(STATUS "Seeta Model Encrypt: OFF")
endif()

if(SEETA_AUTHORIZE)
	message(STATUS "SeetaAuthorize: " ${SEETA_AUTHORIZE})
elseif()
endif ()

if (SEETA_MODEL_ENCRYPT)
   message(STATUS "SeetaModelEncrypt: " ${SEETA_MODEL_ENCRYPT})
endif ()

if ("${PLATFORM}" STREQUAL "x86")
    message(STATUS "Platform: " ${PLATFORM})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
elseif ("${PLATFORM}" STREQUAL "x64")
    message(STATUS "Platform: " ${PLATFORM})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
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

add_subdirectory(${SOLUTION_DIR}/QualityAssessor)
#add_subdirectory(${SOLUTION_DIR}/example)
