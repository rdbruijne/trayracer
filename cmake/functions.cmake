# get directory for file
function(get_relative_directory filePath relativeDir)
	if (IS_ABSOLUTE "${filePath}")
		file(RELATIVE_PATH relSourcePath "${CMAKE_CURRENT_SOURCE_DIR}" "${filePath}")
	else()
		set(relSourcePath "${filePath}")
	endif()
	get_filename_component(pathForFile "${relSourcePath}" DIRECTORY)
	string(REPLACE "\\" "/" relDir "${pathForFile}")
	set("${relativeDir}" "${relDir}" PARENT_SCOPE)
endfunction()



# filters for source files
function(assign_source_group)
    foreach(f ${ARGN})
		get_relative_directory("${f}" dir)
		source_group("${dir}" FILES "${f}")
    endforeach()
endfunction()



# copy multiple files
function(copy_files_post targetProject targetDirectory)
	string(LENGTH ${CMAKE_SOURCE_DIR} sourceDirLength)
	math(EXPR sourceDirLength "${sourceDirLength}+1")
	string(SUBSTRING ${targetDirectory} ${sourceDirLength} -1 target2)
	foreach(f ${ARGN})
		string(SUBSTRING ${f} ${sourceDirLength} -1 f2)
		add_custom_command(
			TARGET ${targetProject}
			POST_BUILD
				#COMMAND ${CMAKE_COMMAND} -E echo "  ${f2} -> ${target2}/"
				COMMAND ${CMAKE_COMMAND} -E make_directory \"${targetDirectory}\"
				COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${f}\" \"${targetDirectory}/\"
		)
	endforeach()
endfunction()



function(copy_files_post_relative targetProject targetDirectory)
	string(LENGTH ${CMAKE_SOURCE_DIR} sourceDirLength)
	math(EXPR sourceDirLength "${sourceDirLength}+1")
	string(SUBSTRING ${targetDirectory} ${sourceDirLength} -1 target2)
	foreach(f ${ARGN})
		string(SUBSTRING ${f} ${sourceDirLength} -1 f2)
		get_relative_directory("${f}" relDir)
		add_custom_command(
			TARGET ${targetProject}
			POST_BUILD
				#COMMAND ${CMAKE_COMMAND} -E echo "  ${f2} -> ${target2}${relDir}/"
				COMMAND ${CMAKE_COMMAND} -E make_directory \"${targetDirectory}${relDir}/\"
				COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${f}\" \"${targetDirectory}${relDir}/\"
		)
	endforeach()
endfunction()



# generate .user file
function(gen_user_file projectName localDebugger commandArgs)
	set(filePath "${CMAKE_CURRENT_BINARY_DIR}/${projectName}.vcxproj.user")
	file(WRITE "${filePath}" "")
	file(APPEND "${filePath}" "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n")
	file(APPEND "${filePath}" "<Project xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n")
	file(APPEND "${filePath}" "  <PropertyGroup>\n")
	file(APPEND "${filePath}" "    <LocalDebuggerCommand>${localDebugger}</LocalDebuggerCommand>\n")
	file(APPEND "${filePath}" "    <LocalDebuggerCommandArguments>${commandArgs}</LocalDebuggerCommandArguments>\n")
	file(APPEND "${filePath}" "    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>\n")
	file(APPEND "${filePath}" "  </PropertyGroup>\n")
	file(APPEND "${filePath}" "</Project>\n")
endfunction()



# make configuration
function(make_configuration name base defines)
	string(TOUPPER ${name} nameUpper)
	
	set("CMAKE_CXX_FLAGS_${nameUpper}"
		"CMAKE_CXX_FLAGS_${base}"
		PARENT_SCOPE
	)
	set("CMAKE_CSharp_FLAGS_${nameUpper}"
		"CMAKE_CSharp_FLAGS_${base}"
		PARENT_SCOPE
	)
	
	set("CMAKE_SHARED_LINKER_FLAGS_${nameUpper}"
		"CMAKE_SHARED_LINKER_FLAGS_${base}"
		PARENT_SCOPE
	)
	
	set(CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} ${name} PARENT_SCOPE)
endfunction()



# get all subdirectories
function(subdirs result curdir)
	file(GLOB children RELATIVE ${curdir} ${curdir}/*)
	set(dirlist "")
	foreach(child ${children})
		if(IS_DIRECTORY ${curdir}/${child})
			list(APPEND dirlist ${child})
		endif()
	endforeach()
	set(${result} ${dirlist} PARENT_SCOPE)
endfunction()



# convert a binary file to a C file
function(bin2c srcFile dstFile)
	string(REGEX MATCH "([^/]+)$" filename ${srcFile})
    string(REGEX REPLACE "\\.| |-" "_" filename ${filename})
    string(TOLOWER ${filename} filename)
    file(READ ${srcFile} filedata HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
	
	foreach(i RANGE 15)
		string(CONCAT foo ${foo} ".....")
	endforeach()
	string(REGEX REPLACE "(${foo})" "\\1\n" filedata ${filedata})
	
    file(WRITE ${dstFile} "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")
	file(APPEND ${dstFile} "static uint8_t ${filename}[] = {\n${filedata}\n};\n")
	file(APPEND ${dstFile} "\n#ifdef __cplusplus\n}\n#endif\n")
endfunction()
