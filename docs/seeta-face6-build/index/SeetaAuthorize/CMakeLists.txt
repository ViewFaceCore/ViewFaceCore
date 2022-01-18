cmake_minimum_required(VERSION 2.7)
project(SeetaAuthorize)
# global root dir
set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DORZ_SOLUTION_DIR="${SOLUTION_DIR}")

# dir for common cmake files
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${SOLUTION_DIR}/cmake)

# option for platform
set(PLATFORM "x64" CACHE STRING "auto, x86 or x64")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Debug or Release")

set(CONFIGURATION ${CMAKE_BUILD_TYPE})

#set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")

# option 
if (MSVC)
set(ORZ_ROOT_DIR "D:/3rd/local" CACHE STRING "The ORZ library")
else()
set(ORZ_ROOT_DIR "/usr/local" CACHE STRING "The ORZ library")
endif()

# set common compiler flags
include(LOCAL_FLAGS)
include(LOCAL_ENV)

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/build CACHE STRING "set install prefix")
install(CODE "MESSAGE(\"Installing into ${CMAKE_INSTALL_PREFIX}\")")

set(LOCAL_OUTPUT_DIR ${PROJECT_SOURCE_DIR})

if (LOCAL_OUTPUT_DIR)
    set(EXECUTABLE_OUTPUT_PATH ${LOCAL_OUTPUT_DIR}/${ENV_RUNTIME_DIR})
    set(LIBRARY_OUTPUT_PATH ${LOCAL_OUTPUT_DIR}/${ENV_LIBRARY_DIR})
endif ()

# build shared lib
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/lock)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/hidden)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/io)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/io)
FILE(GLOB_RECURSE INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

FILE(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

FILE(GLOB_RECURSE SRC_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

add_library(${PROJECT_NAME} SHARED ${INCLUDE_FILES} ${SRC_INCLUDE_FILES} ${SRC_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})

message(STATUS "Build with ORZ")
message(STATUS ${ORZ_ROOT_DIR})
include_directories(${ORZ_ROOT_DIR}/${ENV_HEADER_DIR})
find_library(LIB_ORZ_PATH ORZ_static${ENV_SUFFIX} ${ORZ_ROOT_DIR}/${ENV_ARCHIVE_DIR})
message(STATUS ${ORZ_ROOT_DIR}/${ENV_ARCHIVE_DIR})
target_link_libraries(${PROJECT_NAME} ${LIB_ORZ_PATH})

# install
install(FILES ${SOLUTION_DIR}/cmake/FindSeetaAuthorize.cmake
        DESTINATION cmake)
		
install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
	LIBRARY DESTINATION ${ENV_LIBRARY_DIR}
	ARCHIVE DESTINATION ${ENV_ARCHIVE_DIR}
)

install(DIRECTORY ${SOLUTION_DIR}/include/
	DESTINATION ${ENV_HEADER_DIR}
)


