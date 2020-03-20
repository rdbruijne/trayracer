#pragma once

// Project
#include "Helpers.h"

// OptiX
#include "optix7/optix_device.h"

struct Payload_Wireframe
{
	int primitveID;
};



extern "C" __global__
void __anyhit__Wireframe()
{
	Generic_AnyHit();
}



extern "C" __global__
void __closesthit__Wireframe()
{
	Payload_Wireframe* p = reinterpret_cast<Payload_Wireframe*>(UnpackPointer(optixGetPayload_0(), optixGetPayload_1()));
	p->primitveID = optixGetPrimitiveIndex();
}



extern "C" __global__
void __miss__Wireframe()
{
	Payload_Wireframe* p = reinterpret_cast<Payload_Wireframe*>(UnpackPointer(optixGetPayload_0(), optixGetPayload_1()));
	p->primitveID = -1;
}



extern "C" __global__
void __raygen__Wireframe()
{
	InitializeFilm();

	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	int seed = ix + (optixLaunchParams.resolutionX * iy) + optixLaunchParams.sampleCount;

	// encode payload pointer
	Payload_Wireframe payload = {};
	uint32_t u0, u1;
	PackPointer(&payload, u0, u1);

	constexpr int wireframeRayCount = 3;
	int primId = -1;
	for(int i = 0; i < wireframeRayCount; i++)
	{
		// ray direction
		float3 rayDir = SampleRay(make_float2(ix, iy), make_float2(optixLaunchParams.resolutionX, optixLaunchParams.resolutionY), make_float2(frand(seed), frand(seed)));

		// trace the ray
		optixTrace(optixLaunchParams.sceneRoot,
				   optixLaunchParams.cameraPos,
				   rayDir,
				   0.f,
				   1e20f,
				   0.f,
				   OptixVisibilityMask(255),
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				   RayType_Surface,
				   RayType_Count,
				   RayType_Surface,
				   u0, u1);

		if(i == 0)
			primId = payload.primitveID;
		else if(primId != payload.primitveID)
			return;
	}

	WriteResult(make_float3(1, 1, 1));
}
