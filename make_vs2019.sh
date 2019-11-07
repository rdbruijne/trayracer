exit_on_error()
{
	errcode=$?
	echo "error $errorcode"
	echo "the command executing at the time of the error was"
	echo "$BASH_COMMAND"
	echo "on line ${BASH_LINENO[0]}"
	exit $errcode
}
trap exit_on_error ERR



pushd .

#rm -rf build

# make directories
if [ ! -d "build" ]; then
	mkdir build
fi
cd build



# call cmake
cmake -G "Visual Studio 16 2019" -A x64 -T v142 -DCMAKE_CUDA_FLAGS="-arch=sm_61" ../
popd
