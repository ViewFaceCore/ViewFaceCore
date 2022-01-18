# test

SET(TEST_PROJECT_NAME example)

file(GLOB_RECURSE TEST_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

add_executable(${TEST_PROJECT_NAME} ${TEST_SRC_FILES})

set(OUT_DIR ${SOLUTION_DIR}/build)

include_directories(${OUT_DIR}/${ENV_INCLUDE_DIR})

set_target_properties(${TEST_PORJECT_NAME} PROPERTIES OUTPUT_NAME ${TEST_PROJECT_NAME}${ENV_SUFFIX})

#set(EXECUTABLE_OUTPUT_PATH ${OUT_DIR}/${ENV_RUNTIME_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${ENV_RUNTIME_DIR})

link_directories(${OUT_DIR}/${ENV_LIBRARY_DIR})

target_link_libraries(${TEST_PROJECT_NAME} ${PROJECT_NAME})

find_package(OpenCV)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDES})
    target_link_libraries(${TEST_PROJECT_NAME} ${OpenCV_LIBRARIES})
else ()
    message(WARNING "Can not find OpenCV libraries, make ${TEST_PROJECT_NAME} would failed.")
endif ()
