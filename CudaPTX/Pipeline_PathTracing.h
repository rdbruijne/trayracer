#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__PathTracing()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__PathTracing()
{
	Generic_ClosestHit();
}



extern "C" __global__
void __miss__PathTracing()
{
	Generic_Miss();
}



extern "C" __global__
void __raygen__PathTracing()
{
	Generic_RayGen();
}
