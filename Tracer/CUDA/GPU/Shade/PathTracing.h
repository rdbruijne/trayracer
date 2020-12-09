#pragma once

#include "CudaUtility.h"

// Closures
#include "BSDF/Diffuse.h"

__global__ void PathTracingKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const float4 D4 = pathStates[jobIdx + (stride * 1)];
	const float4 T4 = pathStates[jobIdx + (stride * 2)];

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const float3 T = make_float3(T4);
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

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
	const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);

	// denoiser data
	if(pathLength == 0)
	{
		albedo[pixelIx] = make_float4(attrib.diffuse, 0);
		normals[pixelIx] = make_float4(attrib.shadingNormal, 0);
	}

	// emissive
	if(attrib.emissive.x + attrib.emissive.y + attrib.emissive.z > Epsilon)
	{
		if(pathLength == 0)
		{
			accumulator[pixelIx] += make_float4(attrib.emissive, 0);
			albedo[pixelIx] = make_float4(attrib.emissive, 0);
		}
		else
		{
			accumulator[pixelIx] += make_float4(T * attrib.emissive, 0);
		}
		return;
	}

	// new throughput
	float3 throughput = T * attrib.diffuse;

	// sample light
	const float3 I = O + D * tmax;
	float lightProb;
	float lightPdf;
	float lightDist;
	float3 lightRadiance;
	const float3 L = SampleLight(seed, I, attrib.shadingNormal, lightProb, lightPdf, lightRadiance, lightDist);
	const float NdotL = dot(L, attrib.shadingNormal);

	// hit -> eye
	const float3 wow = -D;
	const float3 wo = normalize(make_float3(dot(wow, attrib.tangent), dot(wow, attrib.bitangent), dot(wow, attrib.geometricNormal)));

	// hit -> light
	const float3 wiw = L;
	const float3 wi = normalize(make_float3(dot(wiw, attrib.tangent), dot(wiw, attrib.bitangent), dot(wiw, attrib.geometricNormal)));

	// sample closure
	ShadingInfo info;
	info.wo   = wo;
	info.wi   = wi;
	info.seed = seed;
	info.T    = throughput;

	Closure closure = Diffuse_Closure(info);
	seed = info.seed;

	// extend ray
	const float3 extend = normalize(closure.extend.wi.x * attrib.tangent + closure.extend.wi.y * attrib.bitangent + closure.extend.wi.z * attrib.geometricNormal);

	// shadow ray
	if(NdotL > 0 && lightPdf > 0)//Epsilon && closure.shadow.pdf > Epsilon)
	{
		// fire shadow ray
		const int32_t shadowIx = atomicAdd(&counters->shadowRays, 1);
		shadowRays[shadowIx + (stride * 0)] = make_float4(I, __int_as_float(pixelIx));
		shadowRays[shadowIx + (stride * 1)] = make_float4(L, lightDist);
		shadowRays[shadowIx + (stride * 2)] = make_float4(closure.shadow.T * lightRadiance * NdotL, 0);
		//shadowRays[shadowIx + (stride * 2)] = make_float4((throughput * lightRadiance * NdotL) / (lightProb * lightPdf), 0);
	}

	// Russian roulette
	if(pathLength > 0)
	{
		const float rr = min(1.0f, max(throughput.x, max(throughput.y, throughput.z)));
		if(rr < rnd(seed))
			return;
		throughput *= 1.f / rr;
	}

	// generate extend
	const float3 newOrigin = O + (D * tmax);
	const float3 newDir = extend;

	// update path states
	const int32_t extendIx = atomicAdd(&counters->extendRays, 1);
	pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __int_as_float(pathIx));
	pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);
	pathStates[extendIx + (stride * 2)] = make_float4(throughput, closure.extend.pdf);
}
