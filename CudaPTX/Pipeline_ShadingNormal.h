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
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();

	// get intersection info
	const int primID = optixGetPrimitiveIndex();
	const float2 uv = optixGetTriangleBarycentrics();
	const float3 D = optixGetWorldRayDirection();

	const uint3 index = meshData.indices[primID];

	// calculate normal at intersection point
	const float3& N0 = meshData.normals[index.x];
	const float3& N1 = meshData.normals[index.y];
	const float3& N2 = meshData.normals[index.z];
	const float3 N = normalize(((1.f - uv.x - uv.y) * N0) + (uv.x * N1) + (uv.y * N2));

	// set payload
	Payload* p = GetPayload();
	p->color = (N + make_float3(1)) * 0.5f;
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
