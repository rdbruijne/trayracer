#pragma once

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// C++
#include <string>

#define CUDA_CHECK(x) CudaCheck((x), __FILE__, __LINE__)

namespace Tracer
{
	void CudaCheck(CUresult res, const char* file, int line);
	void CudaCheck(cudaError_t res, const char* file, int line);

	std::string ToString(CUresult cuResult);
}
