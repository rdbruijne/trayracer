#pragma once

// Project
#include "Helpers.h"

// CUDA
#include "CUDA/random.h"



extern "C" __global__
void __anyhit__DiffuseFilter()
{
	optixTerminateRay();
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
		const uint3 index = meshData.indices[primID];

		// calculate texcoord at intersection point
		const float3 texcoord = Barycentric(optixGetTriangleBarycentrics(), meshData.texcoords[index.x], meshData.texcoords[index.y], meshData.texcoords[index.z]);

		// sample texture
		const float4 diffMap = tex2D<float4>(meshData.diffuseMap, texcoord.x, texcoord.y);
		diff *= make_float3(diffMap.z, diffMap.y, diffMap.x);
	}

	WriteResult(diff);
}



extern "C" __global__
void __miss__DiffuseFilter()
{
	WriteResult(make_float3(0));
}



extern "C" __global__
void __raygen__DiffuseFilter()
{
	InitializeFilm();

	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	// set the seed
	uint32_t seed = tea<2>(ix + (optixLaunchParams.resolutionX * iy), optixLaunchParams.sampleCount);

	// trace the ray
	float3 rayDir = SampleRay(make_float2(ix, iy), make_float2(optixLaunchParams.resolutionX, optixLaunchParams.resolutionY), make_float2(rnd(seed), rnd(seed)));
	optixTrace(optixLaunchParams.sceneRoot, optixLaunchParams.cameraPos, rayDir, 0.f, 1e20f, 0.f, OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface);
}
