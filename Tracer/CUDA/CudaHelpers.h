#pragma once

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// C++
#include <cassert>
#include <string>
#include <sstream>



/*!
 * @brief Check the value of a CUresult. Will throw an exception when an error occurs.
 */
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



/*!
 * @brief Check the value of a cudaError_t. Will throw an exception when an error occurs.
 */
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
	/*!
	* @brief Convert CUresult to corresponding string.
	* @param[in] cuResult CUresult code to convert.
	* @return String containing error code.
	*/
	std::string ToString(CUresult cuResult);
}
