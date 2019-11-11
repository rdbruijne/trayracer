# Gather source files
file(GLOB_RECURSE Tracer_SRC
	CONFIGURE_DEPENDS
	"*.h"
	"*.cpp"
	"*.c"
	"*.inl"
	"*.o"
)
assign_source_group(${Tracer_SRC})



# Declare project
add_executable(Tracer ${Tracer_SRC})
target_compile_definitions(Tracer PRIVATE "_CRT_SECURE_NO_WARNINGS;GLEW_STATIC")
set_target_properties(Tracer PROPERTIES LINK_FLAGS "/NODEFAULTLIB:LIBCMT")

# Include directories
subdirs(TracerSubdirs ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(Tracer
	.
	${TracerSubdirs}
	"${LIBRARIES_DIR}"
	"${LIBRARIES_DIR}/glew"
	"${LIBRARIES_DIR}/glfw"
	"${LIBRARIES_DIR}/optix7"
)



# Link libraries
target_link_libraries(Tracer
	"cudart.lib"
	"opengl32.lib"
	"${LIBRARIES_DIR}/glew/glew32s.lib"
	"${LIBRARIES_DIR}/glfw/glfw3.lib"
)



# set working directory
set_property(TARGET Tracer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
