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
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();

	// get intersection info
	const int primID = optixGetPrimitiveIndex();
	const float2 uv = optixGetTriangleBarycentrics();

	const uint3 index = meshData.indices[primID];

	// calculate texcoord at intersection point
	const float3& T0 = meshData.texcoords[index.x];
	const float3& T1 = meshData.texcoords[index.y];
	const float3& T2 = meshData.texcoords[index.z];
	const float3 T = normalize(((1.f - uv.x - uv.y) * T0) + (uv.x * T1) + (uv.y * T2));

	// set payload
	Payload* p = GetPayload();
	p->color = make_float3(T.x, T.y, 0);
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
