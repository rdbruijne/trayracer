file(GLOB_RECURSE ResourceFiles "*.*")
add_custom_target(Resources)
set_target_properties(Resources PROPERTIES FOLDER Utility)
copy_files_post_relative(Resources "${OUTPUT_DIR}/" ${ResourceFiles})
