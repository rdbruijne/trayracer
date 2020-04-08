#pragma once

// Project
#include "Helpers.h"

// CUDA
#include "CUDA/random.h"



extern "C" __global__
void __anyhit__PathTracing()
{
	optixTerminateRay();
}



extern "C" __global__
void __closesthit__PathTracing()
{
	IntersectionAttributes attrib = GetIntersectionAttributes();

	// basic NdotL
	const float3 L = normalize(make_float3(1, 1, 1));
	const float ndotl = fabsf(dot(attrib.shadingNormal, L));

	// diffuse
	float3 diff = attrib.meshData->diffuse;
	if(attrib.meshData->textures & Texture_DiffuseMap)
	{
		const float4 diffMap = tex2D<float4>(attrib.meshData->diffuseMap, attrib.texcoord.x, attrib.texcoord.y);
		diff *= make_float3(diffMap.z, diffMap.y, diffMap.x);
	}

	// set payload
	Payload* p = GetPayload();
	p->throughput = diff * (.2f + .8f * ndotl);
}



extern "C" __global__
void __miss__PathTracing()
{
	Payload* p = GetPayload();
	p->dst = optixGetRayTmax();
	p->status = RS_Sky;
	WriteResult(make_float3(0));
}



extern "C" __global__
void __raygen__PathTracing()
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

	// trace the ray
	float3 rayDir = SampleRay(make_float2(ix, iy), make_float2(optixLaunchParams.resolutionX, optixLaunchParams.resolutionY), make_float2(rnd(seed), rnd(seed)));
	optixTrace(optixLaunchParams.sceneRoot, optixLaunchParams.cameraPos, rayDir, 0.f, 1e20f, 0.f, OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface, u0, u1);

	if(payload.status == RS_Active)
		WriteResult(payload.throughput);
}
