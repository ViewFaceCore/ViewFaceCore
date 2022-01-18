# library in subdir

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_definitions(-DSEETA_EXPORTS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE ROOT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    )
	
	
file(GLOB_RECURSE COMMON_INCLUDE_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/Common/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/Common/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/Common/*.hpp
	)

file(GLOB_RECURSE ROOT_SEETA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cpp
    )

file(GLOB INCLUDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.hpp
    )

	
list(APPEND SRC_FILES ${ROOT_SRC_FILES})
list(APPEND SRC_FILES ${ROOT_SEETA_FILES})

# add library
add_library(${PROJECT_NAME} ${TARGET} ${SRC_FILES})

find_package(SeetaFaceDetector REQUIRED)
include_directories(${SeetaFaceDetector_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${SeetaFaceDetector_LIBRARIES})
# link ORZ
target_link_libraries(${PROJECT_NAME} ORZ_static${ENV_SUFFIX})

# set library properties
if ("${CONFIGURATION}" STREQUAL "Debug")
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME}d)
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
endif()

# build dir
set(OUT_DIR ${SOLUTION_DIR}/build)
#set(OUT_DIR ${SEETA_INSTALL_PATH}/${SeetaFaceTracking_NAME})

#create install dir
file(MAKE_DIRECTORY ${OUT_DIR})

set(LIBRARY_OUTPUT_PATH ${OUT_DIR}/${ENV_LIBRARY_DIR})
set(ARCHIVE_OUTPUT_PATH ${OUT_DIR}/${ENV_LIBRARY_DIR})
# add command to copy header files
add_custom_command(TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}
        )
add_custom_command(TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
        )

add_custom_command(TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta/Common
        )
# copy every header files
foreach(header ${INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
            )
endforeach()
foreach(header ${COMMON_INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta/Common
            )
endforeach()

#install
#update and install cmake file
configure_file(${SOLUTION_DIR}/cmake/FindSeetaFaceTracking.cmake.in ${SOLUTION_DIR}/cmake/FindSeetaFaceTracking.cmake @ONLY)
file(GLOB_RECURSE FIND_CMAKE_FILES
    ${SOLUTION_DIR}/cmake/FindSeetaFaceTracking.cmake
    )
install(FILES ${FIND_CMAKE_FILES} DESTINATION cmake)	

message(STATUS "install targets library: " ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
	LIBRARY DESTINATION ${ENV_LIBRARY_DIR}
	ARCHIVE DESTINATION ${ENV_ARCHIVE_DIR}
)

message("destination:" ${CMAKE_INSTALL_PREFIX})
install(FILES ${COMMON_INCLUDE_FILES} DESTINATION ${ENV_INCLUDE_DIR}/seeta/Common)
install(FILES ${INCLUDE_FILES} DESTINATION ${ENV_INCLUDE_DIR}/seeta)
