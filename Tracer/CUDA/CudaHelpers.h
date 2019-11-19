#pragma once

// Project
#include "Utility.h"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// C++
#include <stdexcept>

namespace Tracer
{
	/*!
	 * @brief Check if a CUDA result is valid.
	 *
	 * Will throw a runtime error if the cudaResult is not `cudaSuccess`.
	 * @param[in] cudaResult The result to check.
	 */
	static inline void Check(cudaError_t cudaResult)
	{
		if (cudaResult != cudaSuccess)
		{
			const std::string msg = format("CUDA error: %s (%s)\n", cudaGetErrorName(cudaResult), cudaGetErrorString(cudaResult));
			throw std::runtime_error(msg.c_str());
		}
	}
}
