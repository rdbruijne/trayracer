#include "CUDA/CudaHelpers.h"

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
		//assert(res == CUDA_SUCCESS);
		if(res != CUDA_SUCCESS)
		{
			const std::string errorMessage = format("CUDA error in \"%s\" @ %d: %s (%d)", file, line, ToString(res).c_str(), res);
			throw std::runtime_error(errorMessage);
		}
	}



	void CudaCheck(cudaError_t res, const char* file, int line)
	{
		//assert(res == cudaSuccess);
		if(res != cudaSuccess)
		{
			const std::string errorMessage = format("CUDA error in \"%s\" @ %d: %s (%s)", file, line, cudaGetErrorName(res), cudaGetErrorString(res));
			throw std::runtime_error(errorMessage);
		}
	}



	std::string Tracer::ToString(CUresult cuResult)
	{
		const char* errorName = nullptr;
		cuGetErrorName(cuResult, &errorName);

		const char* errorString = nullptr;
		cuGetErrorString(cuResult, &errorString);

		return std::string(errorName) + ": " + errorString;
	}
}
