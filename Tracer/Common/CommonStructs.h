#pragma once

// Project
#include "Optix/Optix7.h"

// CUDA
#include <vector_types.h>

// C++
#include <stdint.h>

struct alignas(16) LaunchParams
{
	float3 cameraPos;
	int32_t resolutionX;

	float3 cameraSide;
	int32_t resolutionY;

	float3 cameraUp;
	int32_t frameID;

	float3 cameraForward;
	int32_t dummy;

	OptixTraversableHandle sceneRoot;
	uint32_t* colorBuffer;
};



enum RayTypes
{
	RAY_TYPE_SURFACE = 0,

	RAY_TYPE_COUNT
};

