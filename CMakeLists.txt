cmake_minimum_required (VERSION 3.9)



# Includes
include("cmake/functions.cmake")



# Define project
get_filename_component(projectDir "${CMAKE_SOURCE_DIR}" NAME)
project("${projectDir}" LANGUAGES CXX CUDA)



# Project settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /debug:fastlink")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /debug:fastlink")
#add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)



# First for the generic no-config case (e.g. with mingw)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})
# Second, for multi-config builds (e.g. msvc)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
	string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${OUTPUT_DIR})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${OUTPUT_DIR})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${OUTPUT_DIR})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)



# Create projcts
add_subdirectory("CudaPTX")
add_subdirectory("Resources")
add_subdirectory("Tracer")



# Set dependencies
add_dependencies(CopyPTX CudaPTX)
add_dependencies(Tracer CopyPTX)
add_dependencies(Tracer Resources)



# Set default startup project
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Tracer)
