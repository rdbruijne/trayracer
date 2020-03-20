#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__AmbientOcclusion()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__AmbientOcclusion()
{
	Generic_ClosestHit();
}



extern "C" __global__
void __miss__AmbientOcclusion()
{
	Generic_Miss();
}



extern "C" __global__
void __raygen__AmbientOcclusion()
{
	Generic_RayGen();
}
