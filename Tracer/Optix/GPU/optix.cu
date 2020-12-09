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
constexpr float DST_MAX = 1e30f;



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
inline void GenerateCameraRay(float3& O, float3& D, int2 pixelIndex, uint32_t& seed)
{
	const float2 index = make_float2(pixelIndex);
	const float2 jitter = make_float2(rnd(seed), rnd(seed));

	const float2 res = make_float2(params.resX, params.resY);
	const float aspect = res.x / res.y;
	float2 screen = (((index + jitter) / res) * 2.0f) - make_float2(1, 1);
	screen.y /= aspect;

	const float tanFov2 = tanf(params.cameraFov / 2.0f);
	const float2 lensCoord = tanFov2 * screen;

	O = params.cameraPos;
	D = normalize(params.cameraForward + (lensCoord.x * params.cameraSide) + (lensCoord.y * params.cameraUp));
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
	case RayGen_Primary:
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
			uint32_t tmax = __float_as_uint(DST_MAX);

			// generate ray
			float3 O, D;
			GenerateCameraRay(O, D, make_int2(ix, iy), seed);

			// trace the ray
			optixTrace(params.sceneRoot, O, D, params.epsilon, DST_MAX, 0.f, 0xFF, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
					   RayType_Surface, RayType_Count, RayType_Surface, bary, instIx, primIx, tmax);

			// set path data
			params.pathStates[pathIx + (stride * 0)] = make_float4(O, __int_as_float(pathIx));
			params.pathStates[pathIx + (stride * 1)] = make_float4(D);
			//params.pathStates[pathIx + (stride * 2)] = make_float4(1, 1, 1, 0);

			// set hit data
			params.hitData[pathIx] = make_uint4(bary, instIx, primIx, tmax);
		}
		break;

	case RayGen_Secondary:
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
				GenerateCameraRay(O, D, make_int2(pixelIx % params.resX, pixelIx / params.resX), seed);
			}

			// prepare the payload
			uint32_t bary = 0;
			uint32_t instIx = ~0u;
			uint32_t primIx = ~0u;
			uint32_t tmax = __float_as_uint(DST_MAX);

			// trace the ray
			optixTrace(params.sceneRoot, O, D, params.epsilon, DST_MAX, 0.f, 0xFF, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
					   RayType_Surface, RayType_Count, RayType_Surface, bary, instIx, primIx, tmax);

			// set hit data
			params.hitData[pathIx] = make_uint4(bary, instIx, primIx, tmax);
		}
		break;

	case RayGen_Shadow:
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

	case RayGen_RayPick:
		{
			// generate ray
			uint32_t seed = 0;
			const int ix = params.rayPickPixel.x;
			const int iy = params.resY - params.rayPickPixel.y;
			float3 O, D;
			GenerateCameraRay(O, D, make_int2(ix, iy), seed);

			// prepare the payload
			uint32_t bary = 0;
			uint32_t instIx = ~0u;
			uint32_t primIx = ~0u;
			uint32_t tmax = __float_as_uint(DST_MAX);

			// trace the ray
			optixTrace(params.sceneRoot, O, D, params.epsilon, DST_MAX, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
					   RayType_Surface, RayType_Count, RayType_Surface, bary, instIx, primIx, tmax);

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
