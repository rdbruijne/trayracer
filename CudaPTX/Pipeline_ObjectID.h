#pragma once

// Project
#include "Helpers.h"

extern "C" __global__
void __anyhit__ObjectID()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__ObjectID()
{
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();
	Payload* p = GetPayload();
	p->color = IdToColor(meshData.objectID + 1);
}



extern "C" __global__
void __miss__ObjectID()
{
	Generic_Miss();
}



extern "C" __global__
void __raygen__ObjectID()
{
	Generic_RayGen();
}
