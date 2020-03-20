#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__ShadingNormal()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__ShadingNormal()
{
	Generic_ClosestHit();
}



extern "C" __global__
void __miss__ShadingNormal()
{
	Payload* p = GetPayload();
	p->color = make_float3(0);
}



extern "C" __global__
void __raygen__ShadingNormal()
{
	Generic_RayGen();
}
