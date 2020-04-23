#pragma once

// Project
#include "Helpers.h"

// CUDA
#include "CUDA/random.h"



extern "C" __global__
void __anyhit__Wireframe()
{
	optixTerminateRay();
}



extern "C" __global__
void __closesthit__Wireframe()
{
	Payload* p = GetPayload();
	p->kernelData.x = optixGetPrimitiveIndex();
}



extern "C" __global__
void __miss__Wireframe()
{
	Payload* p = GetPayload();
	p->status = RS_Sky;
	p->kernelData.x = -1;
}



extern "C" __global__
void __raygen__Wireframe()
{
	InitializeFilm();

	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	// set the seed
	uint32_t seed = tea<2>(ix + (params.resX * iy), params.sampleCount);

	// setup payload
	Payload payload;
	payload.throughput = make_float3(1);
	payload.depth = 0;
	payload.seed = seed;
	payload.status = RS_Active;

	// encode payload pointer
	uint32_t u0, u1;
	PackPointer(&payload, u0, u1);

	// fire several rays, check if their object indices match
	int primId = -1;
	constexpr int wireframeRayCount = 3;
	for(int i = 0; i < wireframeRayCount; i++)
	{
		// generate ray
		float3 O, D;
		GenerateCameraRay(O, D, make_int2(ix, iy), seed);

		// trace the ray
		optixTrace(params.sceneRoot, O, D, params.epsilon, DST_MAX, 0.f, OptixVisibilityMask(255),
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface, u0, u1);

		// check the primitive ID
		if(i == 0)
			primId = payload.kernelData.x;
		else if(primId != payload.kernelData.x)
			return;
	}

	WriteResult(make_float3(1, 1, 1));
}
