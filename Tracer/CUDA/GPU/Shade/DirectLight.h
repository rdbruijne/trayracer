#pragma once

#include "CudaUtility.h"

__global__ void DirectLightKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const float4 D4 = pathStates[jobIdx + (stride * 1)];

	// extract path data
	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	// didn't hit anything
	if(primIx == ~0)
		return;

	// generate seed
	uint32_t seed = tea<2>(pathIx, Params->sampleCount + pathLength + 1);

	// fetch intersection info
	Intersection intersection = {};
	HitMaterial hitMaterial = {};
	GetIntersectionAttributes(instIx, primIx, bary, intersection, hitMaterial);

	// emissive
	if(hitMaterial.emissive.x + hitMaterial.emissive.y + hitMaterial.emissive.z > Epsilon)
	{
		// accounted for in Next Event
		accumulator[pixelIx] += make_float4(hitMaterial.emissive);
		return;
	}

	// new throughput
	const float3 throughput = hitMaterial.diffuse;

	// next event
	const float3 I = O + D * tmax;
	float lightProb;
	float lightPdf;
	float lightDist;
	float3 lightRadiance;
	const float3 L = SampleLight(seed, I, intersection.shadingNormal, lightProb, lightPdf, lightRadiance, lightDist);
	const float NdotL = dot(L, intersection.shadingNormal);
	if(NdotL > 0 && lightPdf > 0)
	{
		// fire shadow ray
		const int32_t shadowIx = atomicAdd(&Counters->shadowRays, 1);
		shadowRays[shadowIx + (stride * 0)] = make_float4(I, __int_as_float(pixelIx));
		shadowRays[shadowIx + (stride * 1)] = make_float4(L, lightDist);
		shadowRays[shadowIx + (stride * 2)] = make_float4(throughput * lightRadiance * NdotL/** (NdotL / (lightProb * lightPdf))*/, 0);
	}

	// denoiser data
	if(pathLength == 0)
	{
		albedo[pixelIx] = make_float4(hitMaterial.diffuse, 0);
		normals[pixelIx] = make_float4(intersection.shadingNormal, 0);
	}
}
