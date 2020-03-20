#pragma once

#include "Common/CommonStructs.h"

// Launch parameters
extern "C" __constant__ LaunchParams optixLaunchParams;



// Ray payload
struct Payload
{
	float3 color;	// 24
	float dummy;	// 8
};
