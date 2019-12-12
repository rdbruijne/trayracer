# https://github.com/robertmaynard/code-samples/blob/master/posts/cmake_ptx/CMakeLists.txt

# Gather source files
file(GLOB_RECURSE GPU_SRC CONFIGURE_DEPENDS
	"program.cu"
)
assign_source_group(${GPU_SRC})



# Declare library
add_library(CudaPTX OBJECT ${GPU_SRC})
set_property(TARGET CudaPTX PROPERTY CUDA_PTX_COMPILATION ON)



# Include directories
include_directories(CudaPTX
	"${CMAKE_SOURCE_DIR}/Tracer/Common"
	"${CMAKE_SOURCE_DIR}/Tracer/Optix"
	"${CMAKE_SOURCE_DIR}/libraries/CUDA"
	"${CMAKE_SOURCE_DIR}/libraries/optix7"
)



# copy ptx
add_custom_target(CopyPTX ALL COMMAND ${CMAKE_COMMAND}
		"-DOBJECTS=$<TARGET_OBJECTS:CudaPTX>"
		"-DOUTPUT=${CMAKE_SOURCE_DIR}/bin/ptx/"
		-P ${CMAKE_SOURCE_DIR}/cmake/copy_objects.cmake
	VERBATIM
	DEPENDS $<TARGET_OBJECTS:CudaPTX>
	COMMENT "Copying ptx files"
)
set_target_properties(CopyPTX PROPERTIES FOLDER Utility)
