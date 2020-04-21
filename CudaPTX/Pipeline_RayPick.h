#pragma once

// Project
#include "Helpers.h"

// CUDA
#include "CUDA/random.h"



extern "C" __global__
void __anyhit__RayPick()
{
	optixTerminateRay();
}



extern "C" __global__
void __closesthit__RayPick()
{
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();

	PayloadRayPick* p = (PayloadRayPick*)GetPayload();
	p->dst = optixGetRayTmax();
	p->objectID = meshData.objectID;
	p->materialID = 0; // #TODO
}



extern "C" __global__
void __miss__RayPick()
{
	PayloadRayPick* p = (PayloadRayPick*)GetPayload();
	p->dst = optixGetRayTmax();
	p->objectID = ~0;
	p->materialID = ~0;
}



extern "C" __global__
void __raygen__RayPick()
{
	// setup payload
	PayloadRayPick payload;

	// encode payload pointer
	uint32_t u0, u1;
	PackPointer(&payload, u0, u1);

	// generate ray
	uint32_t seed = 0;
	const int ix = optixLaunchParams.rayPickPixelIndex.x;
	const int iy = optixLaunchParams.resolutionY - optixLaunchParams.rayPickPixelIndex.y;
	float3 O, D;
	GenerateCameraRay(O, D, make_int2(ix, iy), seed);

	// trace the ray
	optixTrace(optixLaunchParams.sceneRoot, O, D, optixLaunchParams.epsilon, 1e20f, 0.f, OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface, u0, u1);

	RayPickResult& r = *optixLaunchParams.rayPickResult;
	r.rayOrigin = optixLaunchParams.cameraPos;
	r.objectID  = payload.objectID;
	r.rayDir    = D;
	r.dst       = payload.dst;
}
