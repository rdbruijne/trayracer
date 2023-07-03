#pragma once

// Closures
#include "BSDF/Diffuse.h"
#include "BSDF/Disney.h"
#include "BSDF/Lambert.h"

__global__ __launch_bounds__(128, 2)
void PathTracingKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIx + (stride * 0)];
	const float4 D4 = pathStates[jobIx + (stride * 1)];
	const float4 T4 = pathStates[jobIx + (stride * 2)];

	// extract path data
	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const float3 T = make_float3(T4);
	const uint32_t data = __float_as_uint(O4.w);
	const uint32_t pathIx = PathIx(data);
	uint32_t flags = Flags(data);
	const float extendPdf = T4.w;
	const uint32_t pixelIx = pathIx % (resX * resY);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	const float3 I = O + D * tmax;

	// didn't hit anything
	if(primIx == ~0)
	{
		const float3 sky = SampleSky(D, Sky->drawSun && (pathLength == 0 || flags & BsdfFlags::Specular));
		//const float3 contrib = SafeColor(T * sky * 1.f / extendPdf);
		const float3 contrib = SafeColor(T * sky);
		accumulator[pixelIx] += make_float4(contrib, 0);

		// denoiser data
		if(pathLength == 0)
		{
			albedo[pixelIx] = make_float4(sky, 0);
			normals[pixelIx] = make_float4(0);
		}

		return;
	}

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
		float3 contrib = make_float3(0);
		if((pathLength == 0) || (flags & BsdfFlags::Specular))
		{
			contrib = hitMaterial.emissive;
		}
		else
		{
			// apply MIS
			//const float3 prevN = UnpackNormal(__float_as_uint(D4.w));
			const float lightPdf = LightPdf(D, tmax, intersection.area, intersection.shadingNormal);
			const float pickProb = LightPickProbability(intersection.area, hitMaterial.emissive);
			if ((extendPdf + lightPdf * pickProb) > 0)
				contrib = T * hitMaterial.diffuse * (1.0f / (extendPdf + lightPdf * pickProb));
		}

		accumulator[pixelIx] += make_float4(SafeColor(contrib), 0);

		// denoiser
		if(pathLength == 0)
		{
			accumulator[pixelIx] += make_float4(hitMaterial.emissive, 0);
			albedo[pixelIx] = make_float4(hitMaterial.emissive, 0);
		}

		return;
	}

	// generate seed
	uint32_t seed = tea<2>(Params->kernelSettings.seed + pathIx, Params->sampleCount + pathLength + 1);

	// sample light
	float lightProb;
	float lightPdf;
	float lightDist;
	float3 lightRadiance;
	const float3 L = SampleLight(seed, I, intersection.shadingNormal, lightProb, lightPdf, lightRadiance, lightDist);

	// hit -> eye
	const float3 wow = -D;
	const float3 wo = normalize(WorldToTangent(wow, intersection.shadingNormal, intersection.tangent, intersection.bitangent));

	// hit -> light
	const float3 wiw = L;
	const float3 wi = normalize(WorldToTangent(wiw, intersection.shadingNormal, intersection.tangent, intersection.bitangent));

	// sample closure
	ShadingInfo info;
	info.wo   = wo;
	info.dst  = tmax;
	info.wi   = wi;
	info.T    = T;

	//Closure closure = DiffuseClosure(info, hitMaterial, rnd(seed), rnd(seed));
	//Closure closure = DisneyClosure(info, hitMaterial, rnd(seed), rnd(seed));
	Closure closure = LambertClosure(info, hitMaterial, rnd(seed), rnd(seed));

	// Next Event Estimation (no NEE for specular bounces)
	if((lightPdf > Epsilon) && (closure.shadow.pdf > Epsilon) && ((closure.extend.flags & BsdfFlags::Specular) == 0))
	{
		const float3 shadowT = (closure.shadow.T * lightRadiance) / (lightProb * lightPdf + closure.shadow.pdf);
		__threadfence();
		const int32_t shadowIx = atomicAdd(&Counters->shadowRays, 1);
		__threadfence();
		shadowRays[shadowIx + (stride * 0)] = make_float4(I, __uint_as_float(pixelIx));
		shadowRays[shadowIx + (stride * 1)] = make_float4(L, lightDist);
		shadowRays[shadowIx + (stride * 2)] = make_float4(SafeColor(shadowT), 0);
	}

	// extend ray
	if(closure.extend.pdf > Epsilon)
	{
		float3 throughput = closure.extend.T / fabsf(closure.extend.pdf);

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
		const float3 newDir = normalize(TangentToWorld(closure.extend.wi, intersection.shadingNormal, intersection.tangent, intersection.bitangent));

		// update flags for next bounce
		if(closure.extend.flags & BsdfFlags::Specular)
			flags |= BsdfFlags::Specular;
		else
			flags &= ~BsdfFlags::Specular;

		// update path states
		__threadfence();
		const int32_t extendIx = atomicAdd(&Counters->extendRays, 1);
		__threadfence();
		pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __uint_as_float(Pack(pathIx, flags)));
		pathStates[extendIx + (stride * 1)] = make_float4(newDir, __uint_as_float(PackNormal(intersection.shadingNormal)));
		pathStates[extendIx + (stride * 2)] = make_float4(fixnan(throughput), closure.extend.pdf);
	}
}
