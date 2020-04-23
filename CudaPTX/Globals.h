#pragma once

#include "Common/CommonStructs.h"

// Launch parameters
__constant__
static LaunchParams params;



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

	float3 O;
	uint32_t seed;

	float3 D;
	float dst;

	int3 kernelData; // kernel specific info
	RayStatus status;
};



struct PayloadRayPick
{
	float dst;
	uint32_t objectID;
	uint32_t materialID;
	uint32_t pad1;
};
