#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__MaterialID()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__MaterialID()
{
	Generic_ClosestHit();
}



extern "C" __global__
void __miss__MaterialID()
{
	Generic_Miss();
}



extern "C" __global__
void __raygen__MaterialID()
{
	Generic_RayGen();
}
