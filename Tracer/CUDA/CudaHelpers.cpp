#include "CUDA/CudaHelpers.h"

std::string Tracer::ToString(CUresult cuResult)
{
	const char* errorName = nullptr;
	cuGetErrorName(cuResult, &errorName);

	const char* errorString = nullptr;
	cuGetErrorString(cuResult, &errorString);

	return std::string(errorName) + ": " + errorString;
}
