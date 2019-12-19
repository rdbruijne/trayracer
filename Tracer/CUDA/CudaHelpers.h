#pragma once

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// C++
#include <cassert>
#include <string>
#include <sstream>



#define CU_CHECK(x)																												\
	{																															\
		const CUresult res = x;																									\
		assert(res == CUDA_SUCCESS);																							\
		if(res != CUDA_SUCCESS)																									\
		{																														\
			std::stringstream ss;																								\
			ss << "CUDA error " << ToString(res) << "(" << res << ")";															\
			throw std::runtime_error(ss.str());																					\
		}																														\
	}



#define CUDA_CHECK(x)																											\
	{																															\
		const cudaError_t res = x;																								\
		assert(res == cudaSuccess);																								\
		if(res != cudaSuccess)																									\
		{																														\
			std::stringstream ss;																								\
			ss << "CUDA error " << cudaGetErrorName(res) << " (" << cudaGetErrorString(res) << ")";								\
			throw std::runtime_error(ss.str());																					\
		}																														\
	}



namespace Tracer
{
	std::string ToString(CUresult cuResult);
}
