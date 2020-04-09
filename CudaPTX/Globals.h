#pragma once

#include "Common/CommonStructs.h"

// Launch parameters
extern "C" __constant__
LaunchParams optixLaunchParams;



enum RayStatus : int
{
	RS_Active = 0,
	RS_Emissive,
	RS_Sky
};



// Ray payload
struct Payload
{
	float3 throughput;
	int depth;

	float3 rayOrigin;
	uint32_t seed;

	float3 rayDir;
	float dst;

	int3 kernelData; // kernel specific info
	RayStatus status;
};



struct PayloadRayPick
{
	float dst;
	int objectID;
	int materialID;
	int pad1;
};
