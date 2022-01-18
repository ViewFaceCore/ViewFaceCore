project(ORZ)

set(ORZ_SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)

option(ORZ_WITH_OPENSSL "Supporting AES, RAS and SSL with OpenSSL" OFF)

include_directories(${ORZ_SOURCE_DIR}/include)
include_directories(${ORZ_SOURCE_DIR}/src)

set(ORZ_MODULES codec io lego mem net sync tools utils)

if (ORZ_WITH_OPENSSL)
    LIST(APPEND ORZ_MODULES ssl)
endif (ORZ_WITH_OPENSSL)

set(ORZ_INCLUDE_FILES)
set(ORZ_SRC_FILES)
set(ORZ_SRC_INCLUDE_FILES)

foreach (module ${ORZ_MODULES})
    FILE(GLOB_RECURSE ORZ_INCLUDE_${module}_FILES ${ORZ_SOURCE_DIR}/include/orz/${module}/*.h)
    LIST(APPEND ORZ_INCLUDE_FILES ${ORZ_INCLUDE_${module}_FILES})

    FILE(GLOB_RECURSE ORZ_SRC_${module}_FILES ${ORZ_SOURCE_DIR}/src/orz/${module}/*.cpp)
    LIST(APPEND ORZ_SRC_FILES ${ORZ_SRC_${module}_FILES})

    FILE(GLOB_RECURSE ORZ_SRC_INCLUDE_${module}_FILES ${ORZ_SOURCE_DIR}/src/orz/${module}/*.h)
    LIST(APPEND ORZ_SRC_INCLUDE_FILES ${ORZ_SRC_INCLUDE_${module}_FILES})
endforeach ()

set(ORZ_INCLUDE_DIR ${ORZ_SOURCE_DIR}/include)

add_library(${PROJECT_NAME}_STATIC STATIC ${ORZ_INCLUDE_FILES} ${ORZ_SRC_INCLUDE_FILES} ${ORZ_SRC_FILES})
# add_library(${PROJECT_NAME}_SHARED SHARED ${ORZ_INCLUDE_FILES} ${ORZ_SRC_INCLUDE_FILES} ${ORZ_SRC_FILES})
# target_link_libraries(${PROJECT_NAME}_SHARED ${PROJECT_NAME}_STATIC)

if (ORZ_WITH_OPENSSL)
    message(STATUS "Build with OpenSSL")
    add_definitions(-DORZ_WITH_OPENSSL)
	find_package(OpenSSL REQUIRED)
    include_directories(${OPENSSL_INCLUDE_DIR})
    # target_link_libraries(${PROJECT_NAME}_SHARED ${OPENSSL_CRYPTO_LIBRARY})
    target_link_libraries(${PROJECT_NAME}_STATIC ${OPENSSL_CRYPTO_LIBRARY})
    # target_link_libraries(${PROJECT_NAME}_SHARED ${OPENSSL_SSL_LIBRARY})
    target_link_libraries(${PROJECT_NAME}_STATIC ${OPENSSL_SSL_LIBRARY})
endif (ORZ_WITH_OPENSSL)

if (WIN32)
    # target_link_libraries(${PROJECT_NAME}_SHARED wininet.lib)
    target_link_libraries(${PROJECT_NAME}_STATIC wininet.lib)
    # target_link_libraries(${PROJECT_NAME}_STATIC ole32.lib)
elseif (APPLE)
    # target_link_libraries(${PROJECT_NAME}_SHARED pthread)
    target_link_libraries(${PROJECT_NAME}_STATIC pthread)
elseif (UNIX)
    # target_link_libraries(${PROJECT_NAME}_SHARED pthread)
    target_link_libraries(${PROJECT_NAME}_STATIC pthread)
    # target_link_libraries(${PROJECT_NAME}_STATIC uuid)
endif ()

# set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})
set_target_properties(${PROJECT_NAME}_STATIC PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_static${ENV_SUFFIX})

if (ORZ_INSTALL)
	install(TARGETS
            # ${PROJECT_NAME}_SHARED
            ${PROJECT_NAME}_STATIC
		RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
		LIBRARY DESTINATION ${ENV_LIBRARY_DIR}
		ARCHIVE DESTINATION ${ENV_ARCHIVE_DIR}
	)
	
    install(DIRECTORY ${ORZ_INCLUDE_DIR}/orz
		DESTINATION ${ENV_HEADER_DIR}
	)

endif (ORZ_INSTALL)
