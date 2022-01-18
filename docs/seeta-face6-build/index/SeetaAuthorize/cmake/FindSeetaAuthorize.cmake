
# find SeetaAuthorize
#<NAME>_FOUND
#<NAME>_INCLUDE_DIRS or <NAME>_INCLUDES
#<NAME>_LIBRARIES or <NAME>_LIBRARIES or <NAME>_LIBS
#<NAME>_VERSION
#<NAME>_DEFINITIONS

#variables:
#<NAME>_NAME
#<NAME>_INCLUDE_DIR
#<NAME>_LIBRARIE

set(SeetaAuthorize_NAME "SeetaAuthorize" CACHE STRING "The seeta authorize library name")
set(SeetaAuthorize_VERSION "" CACHE STRING "The seeta authorize library version")
#set(SEETA_AUTHORIZE_FULL_NAME ${SeetaAuthorize_NAME}${SeetaAuthorize_VERSION}${ENV_SUFFIX} CACHE STRING "The SeetaAuthorize library full name")
set(SeetaAuthorize_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../")
message(STATUS "SEETA_AUTHORIZE_HOME: " ${SeetaAuthorize_MODULE_PATH})
#message(STATUS "SEETA_AUTHORIZE_HOME: " ${SeetaAuthorize_MODULE_PATH})
if(BUILD_ANDROID)
	# if(SeetaAuthorize_ROOT_DIR STREQUAL "")
		# message(SEND_ERROR "Set the path to SeetaAuthorize root folder in the system variable SeetaAuthorize_ROOT_DIR ")
	# endif()
	message(STATUS "find ENV_LIBRARY_DIR: " ${ENV_LIBRARY_DIR})
	set(SeetaAuthorize_INCLUDE_DIR "${SeetaAuthorize_MODULE_PATH}include")
	file(GLOB_RECURSE INCLUDE_FILE
		${SeetaAuthorize_INCLUDE_DIR}/SeetaLANLock.h)
	if("${INCLUDE_FILE}" STREQUAL "")
		set(SeetaAuthorize_INCLUDE_DIR "${SeetaAuthorize_ROOT_DIR}/include")
	endif()
	message(STATUS "SeetaAuthorize include dir : ${SeetaAuthorize_INCLUDE_DIR}")
	file(GLOB SEETA_AUTHORIZE_LIBRARY_DEBUG
		${SeetaAuthorize_MODULE_PATH}${ENV_LIBRARY_DIR}/*${SeetaAuthorize_NAME}*d.a)
	if("${SEETA_AUTHORIZE_LIBRARY_DEBUG}" STREQUAL "")
		file(GLOB SEETA_AUTHORIZE_LIBRARY_DEBUG
		${SeetaAuthorize_ROOT_DIR}/${ENV_LIBRARY_DIR}/*${SeetaAuthorize_NAME}*d.a)
	endif()
	file(GLOB SEETA_AUTHORIZE_LIBRARY_RELEASE
		${SeetaAuthorize_MODULE_PATH}${ENV_LIBRARY_DIR}/*${SeetaAuthorize_NAME}*.a)
	if("${SEETA_AUTHORIZE_LIBRARY_RELEASE}" STREQUAL "")
		file(GLOB SEETA_AUTHORIZE_LIBRARY_RELEASE
		${SeetaAuthorize_ROOT_DIR}/${ENV_LIBRARY_DIR}/*${SeetaAuthorize_NAME}*.a)
	endif()
else()
	find_path(SeetaAuthorize_INCLUDE_DIR
	  NAMES
		SeetaLANLock.h SeetaLockVerifyLAN.h SeetaLockFunction.h model_helper.h 
	  PATHS
		${SeetaAuthorize_ROOT_DIR}
		${SeetaAuthorize_MODULE_PATH}
		ENV SeetaAuthorize_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_HEADER_DIR})

	find_library(SEETA_AUTHORIZE_LIBRARY_DEBUG
	  NAMES 
		${SeetaAuthorize_NAME}${SeetaAuthorize_VERSION}d
	  PATHS
		${SeetaAuthorize_ROOT_DIR}
		${SeetaAuthorize_MODULE_PATH}
		ENV SeetaAuthorize_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_LIBRARY_DIR})

	find_library(SEETA_AUTHORIZE_LIBRARY_RELEASE
	  NAMES 
		${SeetaAuthorize_NAME}${SeetaAuthorize_VERSION}
	  PATHS
		${SeetaAuthorize_ROOT_DIR}
		${SeetaAuthorize_MODULE_PATH}
		ENV SeetaAuthorize_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_LIBRARY_DIR})
endif()	

if ("${CONFIGURATION}" STREQUAL "Debug")
	set(SeetaAuthorize_LIBRARY ${SEETA_AUTHORIZE_LIBRARY_DEBUG})
else()
	set(SeetaAuthorize_LIBRARY ${SEETA_AUTHORIZE_LIBRARY_RELEASE})
endif()
message(STATUS "SeetaAuthorize_LIBRARY : " ${SeetaAuthorize_LIBRARY})
find_package(PackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(${SeetaAuthorize_NAME}
	FOUND_VAR
		SeetaAuthorize_FOUND
	REQUIRED_VARS
		SeetaAuthorize_INCLUDE_DIR
		SeetaAuthorize_LIBRARY
	FAIL_MESSAGE
		"Could not find seeta authorize!try to set the path to SeetaAuthorize root folder in the system variable SeetaAuthorize_ROOT_DIR "
)

if(SeetaAuthorize_FOUND)
	set(SeetaAuthorize_LIBRARIES ${SeetaAuthorize_LIBRARY})
	set(SeetaAuthorize_INCLUDE_DIRS ${SeetaAuthorize_INCLUDE_DIR})
endif()

message(STATUS "SEETA_AUTHORIZE_FOUND: " ${SeetaAuthorize_FOUND})


foreach (inc ${SeetaAuthorize_INCLUDE_DIRS})
    message(STATUS "SeetaAuthorize include: " ${inc})
endforeach ()
foreach (lib ${SeetaAuthorize_LIBRARIES})
    message(STATUS "SeetaAuthorize library: " ${lib})
endforeach ()

