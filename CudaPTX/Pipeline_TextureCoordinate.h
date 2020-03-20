#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__TextureCoordinate()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__TextureCoordinate()
{
	Generic_ClosestHit();
}



extern "C" __global__
void __miss__TextureCoordinate()
{
	Generic_Miss();
}



extern "C" __global__
void __raygen__TextureCoordinate()
{
	Generic_RayGen();
}
