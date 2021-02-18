#pragma once

#include "CudaUtility.h"

// Closures
//#include "BSDF/Diffuse.h"
#include "BSDF/Disney.h"

__global__ void PathTracingKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const float4 D4 = pathStates[jobIdx + (stride * 1)];
	const float4 T4 = pathStates[jobIdx + (stride * 2)];

	// extract path data
	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const float3 T = make_float3(T4);
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t onSensor = __float_as_int(D4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	// didn't hit anything
	if(primIx == ~0)
	{
		if((pathLength > 0) || (pathLength == 0 && onSensor))
			accumulator[pixelIx] += make_float4(T * SampleSky(D, skyData->drawSun && pathLength == 0));
		return;
	}

	// generate seed
	uint32_t seed = tea<2>(pathIx, params->sampleCount + pathLength + 1);

	// fetch intersection info
	Intersection intersection = {};
	HitMaterial hitMaterial = {};
	GetIntersectionAttributes(instIx, primIx, bary, intersection, hitMaterial);

	// denoiser data
	if(pathLength == 0)
	{
		albedo[pixelIx] = make_float4(hitMaterial.diffuse, 0);
		normals[pixelIx] = make_float4(intersection.shadingNormal, 0);
	}

	// emissive
	if(hitMaterial.emissive.x + hitMaterial.emissive.y + hitMaterial.emissive.z > Epsilon)
	{
		// light contribution is already accounted for with Next Event Estimation
		if(pathLength == 0)
		{
			accumulator[pixelIx] += make_float4(hitMaterial.emissive, 0);
			albedo[pixelIx] = make_float4(hitMaterial.emissive, 0);
		}
		return;
	}

	// new throughput
	float3 throughput = T * hitMaterial.diffuse;

	// sample light
	const float3 I = O + D * tmax;
	float lightProb;
	float lightPdf;
	float lightDist;
	float3 lightRadiance;
	const float3 L = SampleLight(seed, I, intersection.shadingNormal, lightProb, lightPdf, lightRadiance, lightDist);
	const float NdotL = dot(L, intersection.shadingNormal);

	// hit -> eye
	const float3 wow = -D;
	const float3 wo = WorldToTangent(wow, intersection.geometricNormal, intersection.tangent, intersection.bitangent);

	// hit -> light
	const float3 wiw = L;
	const float3 wi = WorldToTangent(wiw, intersection.geometricNormal, intersection.tangent, intersection.bitangent);

	// sample closure
	ShadingInfo info;
	info.wo   = wo;
	info.dst  = tmax;
	info.wi   = wi;
	info.T    = throughput;

	//Closure closure = DiffuseClosure(info, hitMaterial, rnd(seed), rnd(seed));
	Closure closure = DisneyClosure(info, hitMaterial, rnd(seed), rnd(seed));

	// shadow ray
	if(NdotL > 0 && lightPdf > Epsilon && closure.shadow.pdf > Epsilon)
	{
		// fire shadow ray
		const float3 contribution = closure.shadow.T * lightRadiance * (NdotL / (lightProb * lightPdf + closure.shadow.pdf));
		const int32_t shadowIx = atomicAdd(&counters->shadowRays, 1);
		shadowRays[shadowIx + (stride * 0)] = make_float4(I, __int_as_float(pixelIx));
		shadowRays[shadowIx + (stride * 1)] = make_float4(L, lightDist);
		shadowRays[shadowIx + (stride * 2)] = make_float4(contribution, 0);
	}

	// extend ray
	if(closure.extend.pdf > Epsilon)
	{
		// Russian roulette
		if(pathLength > 0)
		{
			const float rr = min(1.0f, max(throughput.x, max(throughput.y, throughput.z)));
			if(rr < rnd(seed))
				return;
			throughput *= 1.f / rr;
		}

		// generate extend
		const float3 extend = normalize(TangentToWorld(closure.extend.wi, intersection.geometricNormal, intersection.tangent, intersection.bitangent));
		const float3 newOrigin = O + (D * tmax);
		const float3 newDir = extend;

		// update path states
		const int32_t extendIx = atomicAdd(&counters->extendRays, 1);
		pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __int_as_float(pathIx));
		pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);
		pathStates[extendIx + (stride * 2)] = make_float4(throughput, closure.extend.pdf);
	}
}
