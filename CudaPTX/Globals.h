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

	int seed;
	RayStatus status;
	float dst;
	int padding;

	int4 kernelData; // kernel specific info
};
