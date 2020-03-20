#pragma once

// Project
#include "Optix/Optix7.h"

// CUDA
#include <vector_types.h>

// C++
#include <stdint.h>

enum RayTypes
{
	RayType_Surface = 0,

	RayType_Count
};



struct alignas(16) LaunchParams
{
	float3 cameraPos;
	int32_t resolutionX;

	float3 cameraSide;
	int32_t resolutionY;

	float3 cameraUp;
	int32_t sampleCount;

	float3 cameraForward;
	float cameraFov;

	OptixTraversableHandle sceneRoot;
	float4* colorBuffer;
};



struct alignas(16) TriangleMeshData
{
	float3 diffuse;
	uint32_t objectID;
};

