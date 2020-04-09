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
	p->objectID = -1;
	p->materialID = -1;
}



extern "C" __global__
void __raygen__RayPick()
{
	// setup payload
	PayloadRayPick payload;

	// encode payload pointer
	uint32_t u0, u1;
	PackPointer(&payload, u0, u1);

	// trace the ray
	float3 rayDir = SampleRay(make_float2(optixLaunchParams.rayPickPixelIndex.x, optixLaunchParams.rayPickPixelIndex.y),
							  make_float2(optixLaunchParams.resolutionX, optixLaunchParams.resolutionY),
							  make_float2(0, 0));
	optixTrace(optixLaunchParams.sceneRoot, optixLaunchParams.cameraPos, rayDir, optixLaunchParams.epsilon, 1e20f, 0.f, OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface, u0, u1);

	RayPickResult& r = *optixLaunchParams.rayPickResult;
	r.rayOrigin = optixLaunchParams.cameraPos;
	r.objectID  = payload.objectID;
	r.rayDir    = rayDir;
	r.dst       = payload.dst;
}
