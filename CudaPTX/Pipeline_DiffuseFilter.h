#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__DiffuseFilter()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__DiffuseFilter()
{
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();
	Payload* p = GetPayload();
	p->color = meshData.diffuse;
}



extern "C" __global__
void __miss__DiffuseFilter()
{
	Generic_Miss();
}



extern "C" __global__
void __raygen__DiffuseFilter()
{
	Generic_RayGen();
}
