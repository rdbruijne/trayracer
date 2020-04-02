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
	const float w = 1.f - (uv.x + uv.y);

	const float3 D = optixGetWorldRayDirection();

	const uint3 index = meshData.indices[primID];

	// calculate normal at intersection point
	const float3& N0 = meshData.normals[index.x];
	const float3& N1 = meshData.normals[index.y];
	const float3& N2 = meshData.normals[index.z];
	const float3 N = normalize((w * N0) + (uv.x * N1) + (uv.y * N2));

	// calculate texcoord at intersection point
	const float3& texcoord0 = meshData.texcoords[index.x];
	const float3& texcoord1 = meshData.texcoords[index.y];
	const float3& texcoord2 = meshData.texcoords[index.z];
	const float3 texcoord = (w * texcoord0) + (uv.x * texcoord1) + (uv.y * texcoord2);

	// basic NdotL
	const float3 L = normalize(make_float3(1, 1, 1));
	const float ndotl = fabsf(dot(N, L));

	// diffuse
	float3 diff = meshData.diffuse;
	if(meshData.textures & Texture_DiffuseMap)
	{
		const float4 diffMap = tex2D<float4>(meshData.diffuseMap, texcoord.x, texcoord.y);
		diff *= make_float3(diffMap.z, diffMap.y, diffMap.x);
	}

	// set payload
	Payload* p = GetPayload();
	p->color = diff * (.2f + .8f * ndotl);
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
