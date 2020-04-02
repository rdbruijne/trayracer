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
	const float w = 1.f - (uv.x + uv.y);

	const uint3 index = meshData.indices[primID];

	// calculate texcoord at intersection point
	const float3& texcoord0 = meshData.texcoords[index.x];
	const float3& texcoord1 = meshData.texcoords[index.y];
	const float3& texcoord2 = meshData.texcoords[index.z];
	const float3 texcoord = (w * texcoord0) + (uv.x * texcoord1) + (uv.y * texcoord2);

	// set payload
	Payload* p = GetPayload();
	p->color = make_float3(texcoord.x, texcoord.y, 0);
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
