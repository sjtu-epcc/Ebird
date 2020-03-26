project(ebird)
cmake_minimum_required(VERSION 2.8)

##########################################################
# set the version number
set(Ebird_VERSION_MAJOR 0)
set(Ebird_VERSION_MINOR 1)

##########################################################
# set project structure
SET(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})
SET(SOURCE_DIR src)
SET(HEADER_DIR include)
SET(BENCHMARK_DIR benchmark)

set(THIRD_LIBS "-lgflags")

##########################################################
#third part package settings
include(cmake/cofig.cmake)
include(cmake/cuda.cmake)
include(cmake/glog.cmake)
##########################################################
## set debug or release mode, default is debug

macro(mark_debug)
    message("Debug mode")
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
    set(CMAKE_BUILD_TYPE Debug)
endmacro()

macro(mark_release)
    message("Release mode")
    add_definitions(-DRELEASE)
    set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_BUILD_TYPE Release)
endmacro()

if (NOT CMAKE_BUILD_TYPE)
    mark_release()
elseif (${CMAKE_BUILD_TYPE} STREQUAL Debug)
    mark_debug()
elseif (${CMAKE_BUILD_TYPE} STREQUAL Release)
    mark_release()
endif ()

##########################################################
# set compiler
SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -W -Wno-unused-parameter")
SET(CMAKE_C_FLAGS " -std=c11")
##########################################################
# find all header files and source files
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIR}/*.cc")
file(GLOB_RECURSE CUDA_FILES "${SOURCE_DIR}/*.cu")
file(GLOB_RECURSE HEADER_FILES "${HEADER_DIR}/*.h")

# include them
include_directories(${HEADER_DIR})
include_directories(${SOURCE_DIR})
include_directories(${SOURCE_DIR})
include_directories(${BENCHMARK_DIR})
include_directories(/usr/include)
include_directories(/usr/local/include)


##########################################################
# add src and benchmarking
add_subdirectory(src)
add_subdirectory(benchmark)

ADD_CUSTOM_TARGET(debug
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
        COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
        )

ADD_CUSTOM_TARGET(release
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
        COMMENT "Switch CMAKE_BUILD_TYPE to Release"
        )
