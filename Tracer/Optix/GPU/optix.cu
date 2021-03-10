#pragma once

// Project
#include "Common/CommonStructs.h"

// CUDA
#include "CUDA/helper_math.h"
#include "CUDA/random.h"



//------------------------------------------------------------------------------------------------------------------------------
// Globals
//------------------------------------------------------------------------------------------------------------------------------
static __constant__ LaunchParams params;
constexpr float DstMax = 1e30f;

// math constants
constexpr float Pi = 3.14159265359f;
constexpr float PiOver2 = 1.57079632679f;
constexpr float PiOver4 = 0.78539816339f;



enum RayTypes
{
	RayType_Surface = 0,
	RayType_Shadow,

	RayType_Count
};



//------------------------------------------------------------------------------------------------------------------------------
// Barycentrics
//------------------------------------------------------------------------------------------------------------------------------
static __device__ uint32_t EncodeBarycentrics(const float2& barycentrics)
{
	const uint32_t bx = static_cast<uint32_t>(barycentrics.x * 65535.f) & 0xFFFF;
	const uint32_t by = static_cast<uint32_t>(barycentrics.y * 65535.f) & 0xFFFF;
	return (bx << 16) | by;
}



//------------------------------------------------------------------------------------------------------------------------------
// Camera ray
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline float2 SampleDisk(uint32_t& seed)
{
	// generate random 2D point in [-1, 1] range
	const float2 lensCoord = make_float2((rnd(seed) * 2.f) - 1.f, (rnd(seed) * 2.f) - 1.f);

	// the center
	if(lensCoord.x == 0 && lensCoord.y == 0)
		return lensCoord;

	// apply concentric mapping
	float theta, r;
	if(fabsf(lensCoord.x) > fabsf(lensCoord.y))
	{
		r = lensCoord.x;
		theta = PiOver4 * (lensCoord.y / lensCoord.x);
	}
	else
	{
		r = lensCoord.y;
		theta = PiOver2 - PiOver4 * (lensCoord.x / lensCoord.y);
	}

	return r * make_float2(cosf(theta), sinf(theta));
}



static __device__
inline float2 SampleTriangle(const float2& a, const float2& b, const float2& c, uint32_t& seed)
{
	const float s1 = rnd(seed);
	const float s2 = rnd(seed);
	const float s1_sqr = sqrtf(s1);
	return (1 - s1_sqr) * a + (s1_sqr * (1 - s2)) * b + (s1_sqr * s2) * c;
}



static __device__
inline float2 SampleBokeh(uint32_t& seed)
{
	// need at least 3 sides
	if(params.cameraBokehSideCount < 3)
		return SampleDisk(seed);

	// select random triangle
	const int index = static_cast<int>(rnd(seed) * params.cameraBokehSideCount);
	const float step = (2.f * Pi) / params.cameraBokehSideCount;

	float sin0, sin1, cos0, cos1;
	__sincosf(params.cameraBokehRotation + (index * step), &sin0, &cos0);
	__sincosf(params.cameraBokehRotation + (index + 1) * step, &sin1, &cos1);

	const float2 c0 = make_float2(0, 0);
	const float2 c1 = make_float2(cos0, sin0);
	const float2 c2 = make_float2(cos1, sin1);

	return SampleTriangle(c0, c1, c2, seed);
}



static __device__
inline bool Distort(float2& lensCoord, float tanFov2, float distortion)
{
	// the center is never distorted
	if(lensCoord.x == 0 && lensCoord.y == 0)
		return true;

	// only distort for positive values
	if(distortion <= 0)
		return true;

	// distort based on distance from center
	const float scale = length(lensCoord);
	const float d = (tanFov2 * distortion) * .5f;
	const float distorted = (scale * distortion * atanf(d)) / d;

	// distortions larger than (Pi / 2) don't hit the sensor
	if(distorted >= 1.57079632679f)
		return false;

	lensCoord *= tanf(distorted) / (scale * distortion);
	return true;
}



static __device__
inline void GenerateCameraRay(float3& O, float3& D, float& T, int2 pixelIndex, uint32_t& seed)
{
	// locations
	const float2 index = make_float2(pixelIndex);
	const float2 jitter = make_float2(rnd(seed), rnd(seed));

	// rescale resolution to (-1, 1) range
	const float2 res = make_float2(params.resX, params.resY);
	const float aspect = res.x / res.y;
	float2 screen = (((index + jitter) / res) * 2.0f) - make_float2(1, 1);
	screen.y /= aspect;

	// fov
	const float tanFov2 = tanf(params.cameraFov / 2.0f);

	// lens coordinate
	const float2 ccdCoord = SampleBokeh(seed) * params.cameraAperture;
	float2 lensCoord = (tanFov2 * screen) - (ccdCoord / params.cameraFocalDist);

	// lens distortion
	if(!Distort(lensCoord, tanFov2, params.cameraDistortion))
		return;

	// generate ray
	D = normalize(params.cameraForward + (lensCoord.x * params.cameraSide) + (lensCoord.y * params.cameraUp));
	O = params.cameraPos + (params.cameraSide * ccdCoord.x) + (params.cameraUp * ccdCoord.y);

	// vignetting
	const float v = max(0.f, dot(D, params.cameraForward));
	const float v2 = v * v;
	T = v2 * v2;
}



//------------------------------------------------------------------------------------------------------------------------------
// Film
//------------------------------------------------------------------------------------------------------------------------------
static __device__
inline void InitializeFilm(int pixelIx)
{
	if(params.sampleCount == 0)
		params.accumulator[pixelIx] = make_float4(0, 0, 0, params.multiSample);
	else
		params.accumulator[pixelIx].w += params.multiSample;
}



//------------------------------------------------------------------------------------------------------------------------------
// Hit
//------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__
void __anyhit__()
{
	optixTerminateRay();
}



extern "C" __global__
void __closesthit__()
{
	optixSetPayload_0(EncodeBarycentrics(optixGetTriangleBarycentrics()));
	optixSetPayload_1(optixGetInstanceIndex());
	optixSetPayload_2(optixGetPrimitiveIndex());
	optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
}



extern "C" __global__
void __miss__()
{
	optixSetPayload_0(0);
}



//------------------------------------------------------------------------------------------------------------------------------
// Raygen
//------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__
void __raygen__()
{
	// get the current pixel index
	const uint3 launchIndex = optixGetLaunchIndex();
	const uint3 launchDims = optixGetLaunchDimensions();
	const uint32_t stride = params.resX * params.resY * params.multiSample;

	switch(params.rayGenMode)
	{
	case RayGenModes::Primary:
		{
			const int pixelIx = launchIndex.x + (launchIndex.y * launchDims.x);
			const int pathIx = pixelIx + (launchIndex.z * launchDims.x * launchDims.y);
			const int ix = launchIndex.x;
			const int iy = launchIndex.y;
			const int sampleIx = launchIndex.z;

			if(sampleIx == 0)
				InitializeFilm(pixelIx);

			// set the seed
			uint32_t seed = tea<2>(pathIx, params.sampleCount << 1);

			// prepare the payload
			uint32_t bary = 0;
			uint32_t instIx = ~0u;
			uint32_t primIx = ~0u;
			uint32_t tmax = __float_as_uint(DstMax);

			// generate ray
			float3 O = params.cameraPos;
			float3 D = params.cameraForward;
			float T = 1.f;
			GenerateCameraRay(O, D, T, make_int2(ix, iy), seed);

			// trace the ray
			if(T > 0)
			{
				optixTrace(params.sceneRoot, O, D, params.epsilon, DstMax, 0.f, 0xFF, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
						   RayType_Surface, RayType_Count, RayType_Surface, bary, instIx, primIx, tmax);
			}

			// set path data
			params.pathStates[pathIx + (stride * 0)] = make_float4(O, __int_as_float(pathIx));
			params.pathStates[pathIx + (stride * 1)] = make_float4(D, 0);
			params.pathStates[pathIx + (stride * 2)] = make_float4(T, T, T, 1.f);

			// set hit data
			params.hitData[pathIx] = make_uint4(bary, instIx, primIx, tmax);
		}
		break;

	case RayGenModes::Secondary:
		{
			const int rayIx = launchIndex.x + (launchIndex.y * launchDims.x);

			// extract info
			const float4 O4 = params.pathStates[rayIx + (stride * 0)];
			const float4 D4 = params.pathStates[rayIx + (stride * 1)];
			//const float4 T4 = params.pathStates[rayIx + (stride * 1)];

			// unpack info
			float3 O = make_float3(O4);
			float3 D = make_float3(D4);
			const int pathIx = __float_as_int(O4.w);

			// generate new ray for wireframe mode
			if(params.renderMode == RenderModes::Wireframe)
			{
				const int32_t pixelIx = pathIx % (params.resX * params.resY);
				uint32_t seed = tea<2>(pathIx, (params.sampleCount << 1) | 1);
				float T;
				GenerateCameraRay(O, D, T, make_int2(pixelIx % params.resX, pixelIx / params.resX), seed);
			}

			// prepare the payload
			uint32_t bary = 0;
			uint32_t instIx = ~0u;
			uint32_t primIx = ~0u;
			uint32_t tmax = __float_as_uint(DstMax);

			// trace the ray
			optixTrace(params.sceneRoot, O, D, params.epsilon, DstMax, 0.f, 0xFF, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
					   RayType_Surface, RayType_Count, RayType_Surface, bary, instIx, primIx, tmax);

			// set hit data
			params.hitData[pathIx] = make_uint4(bary, instIx, primIx, tmax);
		}
		break;

	case RayGenModes::Shadow:
		{
			const int rayIx = launchIndex.x + (launchIndex.y * launchDims.x);

			// extract info
			const float4 O4 = params.shadowRays[rayIx + (stride * 0)];
			const float4 D4 = params.shadowRays[rayIx + (stride * 1)];

			const float3 O = make_float3(O4);
			const float3 D = make_float3(D4);
			const int pathIx = __float_as_int(O4.w);

			// prepare the payload
			uint32_t u0 = ~0u;
			uint32_t u1, u2, u3;

			// trace the ray
			optixTrace(params.sceneRoot, O, D, params.epsilon, D4.w - (2 * params.epsilon), 0.f, 0xFF,
					   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
					   RayType_Surface, RayType_Count, RayType_Surface, u0, u1, u2, u3);

			if(!u0)
			{
				const float4 T4 = params.shadowRays[rayIx + (stride * 2)];
				const int pixelIx = __float_as_int(O4.w);
				params.accumulator[pixelIx] += make_float4(T4.x, T4.y, T4.z, 0);
			}
		}
		break;

	case RayGenModes::RayPick:
		{
			// generate ray
			uint32_t seed = 0;
			const int ix = params.rayPickPixel.x;
			const int iy = params.resY - params.rayPickPixel.y;
			float3 O = params.cameraPos;
			float3 D = params.cameraForward;
			float T = 0;
			GenerateCameraRay(O, D, T, make_int2(ix, iy), seed);

			// prepare the payload
			uint32_t bary = 0;
			uint32_t instIx = ~0u;
			uint32_t primIx = ~0u;
			uint32_t tmax = __float_as_uint(DstMax);

			// trace the ray
			if(T > 0)
			{
				optixTrace(params.sceneRoot, O, D, params.epsilon, DstMax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
						   RayType_Surface, RayType_Count, RayType_Surface, bary, instIx, primIx, tmax);
			}

			// fill result
			RayPickResult& r = *params.rayPickResult;
			r.rayOrigin = params.cameraPos;
			r.instIx    = instIx;
			r.rayDir    = D;
			r.tmax      = __uint_as_float(tmax);
			r.primIx    = primIx;
		}
		break;
	};
}
