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
	float3 diff = meshData.diffuse;
	if(meshData.textures & Texture_DiffuseMap)
	{
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
		const float4 diffMap = tex2D<float4>(meshData.diffuseMap, texcoord.x, texcoord.y);
		diff *= make_float3(diffMap.z, diffMap.y, diffMap.x);
	}

	// set payload
	Payload* p = GetPayload();
	p->color = diff;
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
