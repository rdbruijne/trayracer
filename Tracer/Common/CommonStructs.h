#pragma once

// CUDA
#include <vector_types.h>

// OptiX
#include "Optix7.h"

// C++
#include <stdint.h>

/*!
 * @brief Program launch parameters.
 */
struct alignas(16) LaunchParams
{
	float3 cameraPos;
	int32_t resolutionX;

	float3 cameraForward;
	int32_t resolutionY;

	float3 cameraSide;
	int32_t frameID;

	float3 cameraUp;
	int32_t dummy;

	OptixTraversableHandle sceneRoot;
	uint32_t* colorBuffer;
};



/*!
 * @brief Ray type identifiers
 */
enum RayTypes
{
	RAY_TYPE_SURFACE,

	RAY_TYPE_COUNT
};

