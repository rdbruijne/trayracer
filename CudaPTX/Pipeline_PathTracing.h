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

	// emissive
	float3 em = attrib.meshData->emissive;
	if(em.x > 0 || em.y > 0 || em.z > 0)
	{
		Payload* p = GetPayload();
		p->dst = optixGetRayTmax();
		p->status = RS_Emissive;
		WriteResult(p->throughput * em);
		return;
	}

	// diffuse
	float3 diff = attrib.meshData->diffuse;
	if(attrib.meshData->textures & Texture_DiffuseMap)
	{
		const float4 diffMap = tex2D<float4>(attrib.meshData->diffuseMap, attrib.texcoord.x, attrib.texcoord.y);
		diff *= make_float3(diffMap.z, diffMap.y, diffMap.x);
	}

	// set payload
	Payload* p = GetPayload();
	p->throughput *= diff;
	p->dst = optixGetRayTmax();
	p->O = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
	p->D = SampleCosineHemisphere(attrib.shadingNormal, rnd(p->seed), rnd(p->seed));
}



extern "C" __global__
void __miss__PathTracing()
{
	Payload* p = GetPayload();
	p->dst = optixGetRayTmax();
	p->status = RS_Sky;
	WriteResult(p->throughput);
}



extern "C" __global__
void __raygen__PathTracing()
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

	// generate ray
	float3 O, D;
	GenerateCameraRay(O, D, make_int2(ix, iy), seed);

	while(payload.depth < params.maxDepth)
	{
		// trace the ray
		optixTrace(params.sceneRoot, O, D, params.epsilon, DST_MAX, 0.f, OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT, RayType_Surface, RayType_Count, RayType_Surface, u0, u1);

		if(payload.status != RS_Active)
			break;

		O = payload.O;
		D = payload.D;
		payload.depth++;
	}
}
