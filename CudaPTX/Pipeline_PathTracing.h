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

	// basic NdotL
	const float3 L = normalize(make_float3(1, 1, 1));
	const float ndotl = fabsf(dot(N, L));

	// set payload
	Payload* p = GetPayload();
	p->color = meshData.diffuse * (.2f + .8f * ndotl);
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
