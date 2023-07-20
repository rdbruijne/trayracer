#!/bin/bash

function update_repo
{
	local dir=$1
	local url=$2

	cd repos
	if [ -d "$dir"]; then
		cd $dir && git pull
	fi
	cd ../
}

# create 'repos' directory
[ -d "repos" ] || mkdir repos
#cd repos

update_repo("glfw", "https://github.com/glfw/glfw.git")
