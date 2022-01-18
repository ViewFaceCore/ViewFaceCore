
# find TenniS
#<NAME>_FOUND
#<NAME>_INCLUDE_DIRS or <NAME>_INCLUDES
#<NAME>_LIBRARIES or <NAME>_LIBRARIES or <NAME>_LIBS
#<NAME>_VERSION
#<NAME>_DEFINITIONS

#variables:
#<NAME>_NAME
#<NAME>_INCLUDE_DIR
#<NAME>_LIBRARIE

set(TenniS_NAME "tennis" CACHE STRING "The TenniS library name")
set(TenniS_VERSION "" CACHE STRING "The TenniS library version")
set(TenniS_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../")
#message(STATUS "SEETANET_HOME: " $ENV{SEETANET_HOME})
message(STATUS "TenniS default module path: ${TenniS_MODULE_PATH}" )
message(STATUS "TenniS_ROOT_DIR: ${TenniS_ROOT_DIR}")
	
if(BUILD_ANDROID)
	# if(TenniS_ROOT_DIR STREQUAL "")
		# message(SEND_ERROR "Set the path to TenniS root folder in the system variable TenniS_ROOT_DIR ")
	# endif()
	set(TenniS_INCLUDE_DIR "${TenniS_MODULE_PATH}include")
	file(GLOB_RECURSE INCLUDE_FILE
		${TenniS_INCLUDE_DIR}/api/tennis.h)
	if("${INCLUDE_FILE}" STREQUAL "")
		set(TenniS_INCLUDE_DIR "${TenniS_ROOT_DIR}/include")
	endif()
	message(STATUS "TenniS include dir : ${TenniS_INCLUDE_DIR}")
	file(GLOB TenniS_LIBRARY_DEBUG
		${TenniS_MODULE_PATH}${ENV_LIBRARY_DIR}/*${TenniS_NAME}*.so)
	if("${TenniS_LIBRARY_DEBUG}" STREQUAL "")
		file(GLOB TenniS_LIBRARY_DEBUG
		${TenniS_ROOT_DIR}/${ENV_LIBRARY_DIR}/*${TenniS_NAME}*.so)
	endif()
	file(GLOB TenniS_LIBRARY_RELEASE
		${TenniS_MODULE_PATH}${ENV_LIBRARY_DIR}/*${TenniS_NAME}*.so)
	if("${TenniS_LIBRARY_RELEASE}" STREQUAL "")
		file(GLOB TenniS_LIBRARY_RELEASE
		${TenniS_ROOT_DIR}/${ENV_LIBRARY_DIR}/*${TenniS_NAME}*.so)
	endif()
else()
	find_path(TenniS_INCLUDE_DIR
	  NAMES
		api/tennis.h
	  PATHS
		${TenniS_ROOT_DIR}
		${TenniS_MODULE_PATH}
		ENV TenniS_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_HEADER_DIR})
	
	if("${TenniS_INCLUDE_DIR}" STREQUAL "TenniS_INCLUDE_DIR-NOTFOUND")
		set(TenniS_INCLUDE_DIR "${TenniS_MODULE_PATH}include")
	endif()
		
	find_library(TenniS_LIBRARY_DEBUG
	  NAMES 
		${TenniS_NAME}${TenniS_VERSION}
	  PATHS
		${TenniS_ROOT_DIR}
		${TenniS_MODULE_PATH}
		ENV TenniS_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_LIBRARY_DIR})
		
	find_library(TenniS_LIBRARY_RELEASE
	  NAMES 
		${TenniS_NAME}${TenniS_VERSION}
	  PATHS
		${TenniS_ROOT_DIR}
		${TenniS_MODULE_PATH}
		ENV TenniS_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_LIBRARY_DIR})	
endif()

if ("${CONFIGURATION}" STREQUAL "Debug")
	set(TenniS_LIBRARY ${TenniS_LIBRARY_DEBUG})
else()
	set(TenniS_LIBRARY ${TenniS_LIBRARY_RELEASE})
endif()

find_package(PackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(${TenniS_NAME}
	FOUND_VAR
		TENNIS_FOUND
	REQUIRED_VARS
		TenniS_INCLUDE_DIR
		TenniS_LIBRARY
	FAIL_MESSAGE
		"Could not find TenniS! Try to set the path to TenniS root folder in the system variable TenniS_ROOT_DIR"
)

if(TENNIS_FOUND)
	set(TenniS_LIBRARIES ${TenniS_LIBRARY})
	set(TenniS_INCLUDE_DIRS ${TenniS_INCLUDE_DIR})
endif()
message(STATUS "TenniS_FOUND: " ${TENNIS_FOUND})


foreach (inc ${TenniS_INCLUDE_DIRS})
    message(STATUS "TenniS include: " ${inc})
endforeach ()
foreach (lib ${TenniS_LIBRARIES})
    message(STATUS "TenniS library: " ${lib})
endforeach ()

