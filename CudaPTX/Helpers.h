#pragma once

// Project
#include "Globals.h"
#include "Common/CommonStructs.h"

// OptiX
#include "optix7/optix_device.h"

// CUDA
#include "CUDA/helper_math.h"



//------------------------------------------------------------------------------------------------------------------------------
// RNG
//------------------------------------------------------------------------------------------------------------------------------
// generate a random number (0, 1)
static __device__
float frand(int& seed)
{
	// http://www.iquilezles.org/www/articles/sfrand/sfrand.htm
	union
	{
		float fres;
		unsigned int ires;
	};

	seed *= 16807;
	ires = ((((unsigned int)seed)>>9 ) | 0x3f800000);
	return fres - 1.0f;
}



//------------------------------------------------------------------------------------------------------------------------------
// Payload
//------------------------------------------------------------------------------------------------------------------------------
// Pack a pointer into 2 unsigned integers.
static __device__
void PackPointer(void* ptr, uint32_t& u1, uint32_t& u2)
{
	const uint64_t u = reinterpret_cast<uint64_t>(ptr);
	u1 = u >> 32;
	u2 = u & 0xFFFFFFFF;
}



// Unpack a pointer from 2 unsigned integers.
static __device__
void* UnpackPointer(uint32_t u1, uint32_t u2)
{
	return reinterpret_cast<void*>((static_cast<uint64_t>(u1) << 32) | u2);
}



// Get ray payload.
static __device__
Payload* GetPayload()
{
	return reinterpret_cast<Payload*>(UnpackPointer(optixGetPayload_0(), optixGetPayload_1()));
}



//------------------------------------------------------------------------------------------------------------------------------
// Colors
//------------------------------------------------------------------------------------------------------------------------------
// Convert an object ID to a color.
static __device__
float3 IdToColor(uint32_t id)
{
	// https://stackoverflow.com/a/9044057
	uint32_t c[3] = { 0, 0, 0 };
	for(uint32_t i = 0; i < 3; i++)
	{
		c[i] = (id >> i) & 0x249249;
		c[i] = ((c[i] << 1) | (c[i] >>  3)) & 0x0C30C3;
		c[i] = ((c[i] << 2) | (c[i] >>  6)) & 0x00F00F;
		c[i] = ((c[i] << 4) | (c[i] >> 12)) & 0x0000FF;
	}

	return make_float3(c[0] * (1.f / 255.f), c[1] * (1.f / 255.f), c[2] * (1.f / 255.f));
}



//------------------------------------------------------------------------------------------------------------------------------
// Ray
//------------------------------------------------------------------------------------------------------------------------------
static __device__
float3 SampleRay(float2 index, float2 dimensions, float2 jitter)
{
	// screen plane position
	const float2 screen = (index + jitter) / dimensions;

	// ray direction
	const float aspect = dimensions.x / dimensions.y;
	const float3 rayDir = normalize(optixLaunchParams.cameraForward +
								((screen.x - 0.5f) * optixLaunchParams.cameraSide * aspect) +
								((screen.y - 0.5f) * optixLaunchParams.cameraUp));

	return rayDir;
}



//------------------------------------------------------------------------------------------------------------------------------
// Film
//------------------------------------------------------------------------------------------------------------------------------
static __device__
void InitializeFilm()
{
	const uint32_t fbIndex = optixGetLaunchIndex().x + optixGetLaunchIndex().y * optixLaunchParams.resolutionX;
	if(optixLaunchParams.sampleCount == 0)
		optixLaunchParams.colorBuffer[fbIndex] = make_float4(0, 0, 0, 1);
	else
		optixLaunchParams.colorBuffer[fbIndex].w++;

}



static __device__
void WriteResult(float3 result)
{
	// write color to the buffer
	const uint32_t fbIndex = optixGetLaunchIndex().x + optixGetLaunchIndex().y * optixLaunchParams.resolutionX;
	optixLaunchParams.colorBuffer[fbIndex] += make_float4(result, 0);
}



//------------------------------------------------------------------------------------------------------------------------------
// Generic programs
//------------------------------------------------------------------------------------------------------------------------------
static __device__
void Generic_AnyHit()
{
	optixTerminateRay();
}



static __device__
void Generic_ClosestHit()
{
	const TriangleMeshData& meshData = *(const TriangleMeshData*)optixGetSbtDataPointer();
	Payload* p = GetPayload();
	p->color = meshData.diffuse;
}



static __device__
void Generic_Miss()
{
	WriteResult(make_float3(0));
}



static __device__
void Generic_RayGen()
{
	InitializeFilm();

	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	int seed = ix + (optixLaunchParams.resolutionX * iy) + optixLaunchParams.sampleCount;

	// encode payload pointer
	Payload payload = {};
	uint32_t u0, u1;
	PackPointer(&payload, u0, u1);

	// trace the ray
	float3 rayDir = SampleRay(make_float2(ix, iy), make_float2(optixLaunchParams.resolutionX, optixLaunchParams.resolutionY), make_float2(frand(seed), frand(seed)));
	optixTrace(optixLaunchParams.sceneRoot,
			   optixLaunchParams.cameraPos,
			   rayDir,
			   0.f,
			   1e20f,
			   0.f,
			   OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			   RayType_Surface,
			   RayType_Count,
			   RayType_Surface,
			   u0, u1);

	WriteResult(payload.color);
}
