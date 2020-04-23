#pragma once

// Project
#include "Helpers.h"

// CUDA
#include "CUDA/random.h"



extern "C" __global__
void __anyhit__ObjectID()
{
	optixTerminateRay();
}



extern "C" __global__
void __closesthit__ObjectID()
{
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();
	WriteResult(IdToColor(meshData.objectID + 1));
}



extern "C" __global__
void __miss__ObjectID()
{
	WriteResult(make_float3(0));
}



extern "C" __global__
void __raygen__ObjectID()
{
	InitializeFilm();

	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	// set the seed
	uint32_t seed = tea<2>(ix + (params.resX * iy), params.sampleCount);

	// generate ray
	float3 O, D;
	GenerateCameraRay(O, D, make_int2(ix, iy), seed);

	// trace the ray
	optixTrace(params.sceneRoot, O, D, params.epsilon, DST_MAX, 0.f, OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface);
}
