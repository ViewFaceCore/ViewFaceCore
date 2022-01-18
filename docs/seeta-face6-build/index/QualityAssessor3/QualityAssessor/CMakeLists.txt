# library in subdir

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_definitions(-DSEETA_EXPORTS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB ROOT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

file(GLOB_RECURSE SEETA_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cpp
    )

file(GLOB_RECURSE ORZ_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/orz/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/orz/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/orz/*.cpp
    )

file(GLOB_RECURSE SEETA_API_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    )

file(GLOB_RECURSE SEETA_INCLUDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.hpp
    )

file(GLOB_RECURSE SEETA_INCLUDE_COMMON_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/Common/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/Common/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/Common/*.hpp
    )
	
list(APPEND SRC_FILES ${SEETA_SRC_FILES})
list(APPEND SRC_FILES ${ORZ_SRC_FILES})
list(APPEND SRC_FILES ${ROOT_SRC_FILES})
list(APPEND SRC_FILES ${SEETA_API_SRC_FILES})


# add library
add_library(${PROJECT_NAME}_SHARED SHARED ${SRC_FILES})

# link SeetaAuthorize
if(SEETA_AUTHORIZE)
	find_package(SeetaAuthorize REQUIRED)
	include_directories(${SeetaAuthorize_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME}_SHARED ${SeetaAuthorize_LIBRARIES})
endif()

if(SEETA_MODEL_ENCRYPT)
	find_package(SeetaAuthorize REQUIRED)
	include_directories(${SeetaAuthorize_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME}_SHARED ${SeetaAuthorize_LIBRARIES})
endif()

# link ORZ
target_link_libraries(${PROJECT_NAME}_SHARED ORZ_static${ENV_SUFFIX})

find_package(SeetaPoseEstimation REQUIRED)
include_directories(${SeetaPoseEstimation_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_SHARED ${SeetaPoseEstimation_LIBRARIES})

find_package(TenniS REQUIRED)
include_directories(${TenniS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_SHARED ${TenniS_LIBRARIES})

# set library properties
if ("${CONFIGURATION}" STREQUAL "Debug")
    set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME}d)
else()
    set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
endif()

# build dir
set(OUT_DIR ${SOLUTION_DIR}/build)

set(LIBRARY_OUTPUT_PATH ${OUT_DIR}/${ENV_LIBRARY_DIR})

# add command to copy header files
add_custom_command(TARGET ${PROJECT_NAME}_SHARED
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}
        )
add_custom_command(TARGET ${PROJECT_NAME}_SHARED
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
        )

# copy every header files
foreach(header ${INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}_SHARED
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/${ENV_INCLUDE_DIR}
            )
endforeach()
foreach(header ${SEETA_INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}_SHARED
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
            )
endforeach()

install(TARGETS ${PROJECT_NAME}_SHARED DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENV_LIBRARY_DIR})
install(FILES ${SEETA_INCLUDE_COMMON_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENV_INCLUDE_DIR}/seeta/Common)
install(FILES ${SEETA_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENV_INCLUDE_DIR}/seeta)
