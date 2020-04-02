#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__ZDepth()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__ZDepth()
{
	Payload* p = GetPayload();
	const float d = optixGetRayTmax();
	p->color = make_float3(clamp(1.f / logf(d), 0.f, 1.f));
}



extern "C" __global__
void __miss__ZDepth()
{
	Payload* p = GetPayload();
	p->color = make_float3(0, 0, 0);
}



extern "C" __global__
void __raygen__ZDepth()
{
	Generic_RayGen();
}
