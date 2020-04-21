#pragma once

// Project
#include "Helpers.h"

// CUDA
#include "CUDA/random.h"



extern "C" __global__
void __anyhit__TextureCoordinate()
{
	optixTerminateRay();
}



extern "C" __global__
void __closesthit__TextureCoordinate()
{
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();

	// get intersection info
	const int primID = optixGetPrimitiveIndex();
	const uint3 index = meshData.indices[primID];
	const float2 barycentrics = optixGetTriangleBarycentrics();
	const float3 texcoord = Barycentric(barycentrics, meshData.texcoords[index.x], meshData.texcoords[index.y], meshData.texcoords[index.z]);

	WriteResult(make_float3(texcoord.x, texcoord.y, 0));
}



extern "C" __global__
void __miss__TextureCoordinate()
{
	WriteResult(make_float3(0));
}



extern "C" __global__
void __raygen__TextureCoordinate()
{
	InitializeFilm();

	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	// set the seed
	uint32_t seed = tea<2>(ix + (optixLaunchParams.resolutionX * iy), optixLaunchParams.sampleCount);

	// generate ray
	float3 O, D;
	GenerateCameraRay(O, D, make_int2(ix, iy), seed);

	// trace the ray
	optixTrace(optixLaunchParams.sceneRoot, O, D, optixLaunchParams.epsilon, 1e20f, 0.f, OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface);
}
