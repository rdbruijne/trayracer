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

# Compile options
target_compile_definitions(Tracer PRIVATE "_CRT_SECURE_NO_WARNINGS;GLEW_STATIC;__STDC_WANT_SECURE_LIB__=1;NOMINMAX")
target_compile_options(Tracer PRIVATE -WX -Wall -MP)
set_target_properties(Tracer PROPERTIES COMPILE_FLAGS "/wd4100 /wd4324 /wd4505 /wd4514 /wd4710 /wd4711 /wd4820 /wd5045 /wd26812")
	#  4100: 'identifier' : unreferenced formal parameter
	#  4324: 'struct_name' : structure was padded due to __declspec(align())
	#  4505: 'function' : unreferenced local function has been removed
	#  4514: 'function' : unreferenced inline function has been removed
	#  4710: 'function' : function not inlined
	#  4711: function 'function' selected for inline expansion
	#  4820: 'bytes' bytes padding added after construct 'member_name'
	#  5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
	# 26812: The enum type %type% is unscoped. Prefer 'enum class' over 'enum'

# Link options
set_target_properties(Tracer PROPERTIES LINK_FLAGS "/NODEFAULTLIB:LIBCMT /NODEFAULTLIB:MSVCRT")

# Include directories
subdirs(TracerSubdirs ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(Tracer
	.
	${TracerSubdirs}
	"${CMAKE_SOURCE_DIR}/libraries"
	"${CMAKE_SOURCE_DIR}/libraries/CUDA"
	"${CMAKE_SOURCE_DIR}/libraries/glew"
	"${CMAKE_SOURCE_DIR}/libraries/glfw"
	"${CMAKE_SOURCE_DIR}/libraries/optix7"
)



# Link libraries
target_link_libraries(Tracer
	"cuda.lib"
	"cudart.lib"
	"opengl32.lib"
	"${CMAKE_SOURCE_DIR}/libraries/glew/glew32s.lib"
	"${CMAKE_SOURCE_DIR}/libraries/glfw/glfw3.lib"
)



# set working directory
set_property(TARGET Tracer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
