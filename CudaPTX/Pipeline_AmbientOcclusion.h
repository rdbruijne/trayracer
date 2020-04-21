#pragma once

// Project
#include "Helpers.h"

// CUDA
#include "CUDA/random.h"



extern "C" __global__
void __anyhit__AmbientOcclusion()
{
	optixTerminateRay();
}



extern "C" __global__
void __closesthit__AmbientOcclusion()
{
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();

	// get intersection info
	const int primID = optixGetPrimitiveIndex();
	const uint3 index = meshData.indices[primID];
	const float2 barycentrics = optixGetTriangleBarycentrics();
	const float3 N = normalize(Barycentric(barycentrics, meshData.normals[index.x], meshData.normals[index.y], meshData.normals[index.z]));

	// set payload
	Payload* p = GetPayload();
	p->dst = optixGetRayTmax();
	p->kernelData.x = float_as_int(N.x);
	p->kernelData.y = float_as_int(N.y);
	p->kernelData.z = float_as_int(N.z);
}



extern "C" __global__
void __miss__AmbientOcclusion()
{
	Payload* p = GetPayload();
	p->dst = optixGetRayTmax();
	p->status = RS_Sky;
}



extern "C" __global__
void __raygen__AmbientOcclusion()
{
	InitializeFilm();

	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	// set the seed
	uint32_t seed = tea<2>(ix + (optixLaunchParams.resolutionX * iy), optixLaunchParams.sampleCount);

	// setup payload
	Payload payload;
	payload.throughput = make_float3(1);
	payload.depth = 0;
	payload.seed = seed;
	payload.status = RS_Active;

	// encode payload pointer
	uint32_t u0, u1;
	PackPointer(&payload, u0, u1);

	// generate ray
	float3 O, D;
	GenerateCameraRay(O, D, make_int2(ix, iy), seed);

	// trace the ray
	optixTrace(optixLaunchParams.sceneRoot, O, D, optixLaunchParams.epsilon, 1e20f, 0.f, OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface, u0, u1);

	if(payload.status == RS_Active)
	{
		// sample hemisphere
		const float3 normal = make_float3(int_as_float(payload.kernelData.x), int_as_float(payload.kernelData.y), int_as_float(payload.kernelData.z));
		const float3 p = SampleCosineHemisphere(normal, rnd(seed), rnd(seed));

		O = optixLaunchParams.cameraPos + (D * payload.dst);
		D = p;

		// trace the bounce ray
		payload.depth++;
		optixTrace(optixLaunchParams.sceneRoot, O, D, optixLaunchParams.epsilon, 1e20f, 0.f, OptixVisibilityMask(255),
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface, u0, u1);

		if(payload.dst > optixLaunchParams.aoDist)
			WriteResult(make_float3(1));
		else
			WriteResult(make_float3(payload.dst / optixLaunchParams.aoDist));
	}
}
