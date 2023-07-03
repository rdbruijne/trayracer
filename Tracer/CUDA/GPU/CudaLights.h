#pragma once

// Project
#include "CudaGlobals.h"
#include "CudaLinearMath.h"
#include "CudaSky.h"

// CUDA
#include "CUDA/random.h"

#define ENABLE_SKY_NEE false

static __inline__ __device__
float LightPdf(const float3& D, float dst, float lightArea, const float3& lightNormal)
{
	return (dst * dst) / (fabsf(dot(D, lightNormal)) * lightArea);
}



static __inline__ __device__
float LightPickProbability(float area, const float3& em)
{
#if ENABLE_SKY_NEE
	const float totalEnergy = Sky->sunEnergy + LightEnergy;
#else
	const float totalEnergy = LightEnergy;
#endif

	// light energy
	const float3 energy = em * area;
	const float lightEnergy = energy.x + energy.y + energy.z;
	return lightEnergy / totalEnergy;
}



static __inline__ __device__
int32_t SelectLight(uint32_t& seed)
{
	const float e = rnd(seed) * LightEnergy;
	int32_t low = 0;
	int32_t high = LightCount - 1;
	while(low <= high)
	{
		const int32_t mid = (low + high) >> 1;
		const LightTriangle& tri = Lights[mid];
		if(e < tri.sumEnergy)
			high = mid;
		else if(e > tri.sumEnergy + tri.energy)
			low = mid + 1;
		else
			return mid;
	}

	// failed to find a light using importance sampling, pick a random one from the array
	// #NOTE: we should never get here!
	return clamp((int)(rnd(seed) * LightCount), 0, LightCount - 1);
}



static __inline__ __device__
float3 SampleLight(uint32_t& seed, const float3& I, const float3& N, float& prob, float& pdf, float3& radiance, float& dist)
{
#if ENABLE_SKY_NEE
	// energy
	const float totalEnergy = Sky->sunEnergy + LightEnergy;

	// check for any energy
	if(totalEnergy == 0)
	{
		prob = 0;
		pdf = 0;
		radiance = make_float3(0);
		return make_float3(1);
	}

	// try to pick the sun
	if(rnd(seed) * totalEnergy <= Sky->sunEnergy)
	{
		prob     = Sky->sunEnergy / totalEnergy;
		pdf      = 1.f;
		radiance = SampleSky(Sky->sunDir, false);
		dist     = 1e20f;
		return Sky->sunDir;
	}
#else
	const float totalEnergy = LightEnergy;

	// check for any energy
	if(totalEnergy == 0)
	{
		prob = 0;
		pdf = 0;
		radiance = make_float3(0);
		dist = 1e20f;
		return make_float3(0, 1, 0);
	}
#endif

	// pick random light
	const int32_t lightIx = SelectLight(seed);
	const LightTriangle& light = Lights[lightIx];

	// select point on light
	const float3 bary = make_float3(rnd(seed), rnd(seed), rnd(seed));
	const float3 pointOnLight = (bary.x * light.V0) + (bary.y * light.V1) + (bary.z * light.V2);

	// sample direction (light -> hit)
	float3 L = I - pointOnLight;
	const float sqDist = dot(L, L);
	L = normalize(L);
	const float LNdotL = dot(light.N, L);

	// set output parameters
	prob = light.energy / totalEnergy;
	pdf = (LNdotL > 0 && dot(N, L)) ? sqDist / (light.area * LNdotL) : 0;
	dist = sqrtf(sqDist);
	radiance = light.radiance;

	// return hit -> light
	return -L;
}
