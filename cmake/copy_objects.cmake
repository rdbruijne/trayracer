foreach(obj ${OBJECTS})
	get_filename_component(obj_name ${obj} NAME_WE)
	file(COPY ${obj} DESTINATION "${OUTPUT}/")
endforeach()
