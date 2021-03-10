#include "CudaError.h"

// Project
#include "Utility/Utility.h"

// C++
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace Tracer
{
	void CudaCheck(CUresult res, const char* file, int line)
	{
		if(res != CUDA_SUCCESS)
			FatalError("CUDA error in \"%s\" @ %d: %s (%d)", file, line, ToString(res).c_str(), res);
	}



	void CudaCheck(cudaError_t res, const char* file, int line)
	{
		if(res != cudaSuccess)
			FatalError("CUDA error in \"%s\" @ %d: %s (%s)", file, line, cudaGetErrorName(res), cudaGetErrorString(res));
	}



	std::string ToString(CUresult cuResult)
	{
		const char* errorName = nullptr;
		cuGetErrorName(cuResult, &errorName);

		const char* errorString = nullptr;
		cuGetErrorString(cuResult, &errorString);

		return std::string(errorName) + ": " + errorString;
	}
}
