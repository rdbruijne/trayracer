#pragma once

// CUDA
#include <vector_types.h>

// C++
#include <stdint.h>

/*!
 * @brief Program launch parameters.
 */
struct LaunchParams
{
	/*! Render buffer resolution */
	int2      resolution;	// 16

	/*! Render buffer */
	uint32_t* colorBuffer;	// 16

	/*! Frame ID */
	int32_t   frameID;		// 8
};
