// OptiX
#include <optix_device.h>

// CUDA
#include <helper_math.h>

// Project
#include "CommonStructs.h"

//------------------------------------------------------------------------------------------------------------------------------
// GLobal data
//------------------------------------------------------------------------------------------------------------------------------
/*!
 * Launch parameters
 */
extern "C" __constant__ LaunchParams optixLaunchParams;



/*!
 * Ray payload
 */
struct Payload
{
	float3 color;	// 24
	float dummy;	// 8
};



//------------------------------------------------------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------------------------------------------------------
/*!
 * @brief Pack a pointer into 2 unsigned integers.
 */
static __device__
void PackPointer(void* ptr, uint32_t& u1, uint32_t& u2)
{
	const uint64_t u = reinterpret_cast<uint64_t>(ptr);
	u1 = u >> 32;
	u2 = u & 0xFFFFFFFF;
}



/*!
 * @brief Unpack a pointer from 2 unsigned integers.
 */
static __device__
void* UnpackPointer(uint32_t u1, uint32_t u2)
{
	return reinterpret_cast<void*>((static_cast<uint64_t>(u1) << 32) | u2);
}



/*!
 * @brief Get ray payload.
 */
static __device__
Payload* GetPayload()
{
	return reinterpret_cast<Payload*>(UnpackPointer(optixGetPayload_0(), optixGetPayload_1()));
}



/*!
 * @brief Convert an object ID to a color.
 */
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
// Tracing
//------------------------------------------------------------------------------------------------------------------------------
/*!
 * @brief Any hit program.
 */
extern "C" __global__ void __anyhit__radiance()
{
}



/*!
 * @brief Closest hit program.
 */
extern "C" __global__ void __closesthit__radiance()
{
	const int primID = optixGetPrimitiveIndex();
	Payload* p = GetPayload();
	p->color = IdToColor(primID);
}



/*!
 * @brief Miss program.
 */
extern "C" __global__ void __miss__radiance()
{
	Payload* p = GetPayload();
	p->color = make_float3(.25f, .5f, 1);
}



/*!
 * @brief Ray gen program.
 */
extern "C" __global__ void __raygen__renderFrame()
{
	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

#if true
	// encode payload pointer
	Payload payload = {};
	uint32_t u0, u1;
	PackPointer(&payload, u0, u1);

	// screen plane position
	const float2 screen = make_float2(ix + 0.5f, iy + 0.5f) / make_float2(optixLaunchParams.resolutionX, optixLaunchParams.resolutionY);

	// ray direction
	float3 rayDir = normalize(optixLaunchParams.cameraForward +
								((screen.x - 0.5f) * optixLaunchParams.cameraSide) +
								((screen.y - 0.5f) * optixLaunchParams.cameraUp));

	// trace the ray
	optixTrace(optixLaunchParams.sceneRoot,
			   optixLaunchParams.cameraPos,
			   rayDir,
			   0.f,
			   1e20f,
			   0.f,
			   OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			   RAY_TYPE_SURFACE,
			   RAY_TYPE_COUNT,
			   RAY_TYPE_SURFACE,
			   u0, u1);

	// generate a color
	const int r = int(255.f * payload.color.x);
	const int g = int(255.f * payload.color.y);
	const int b = int(255.f * payload.color.z);
	const uint32_t rgba = 0xFF000000 | (r << 0) | (g << 8) | (b << 16);
#else
	// generate a color
	const int r = (ix + optixLaunchParams.frameID) & 255;
	const int g = (iy + optixLaunchParams.frameID * 2) & 255;
	const int b = (ix + iy) & 255;
	const uint32_t rgba = 0xFF000000 | (r << 0) | (g << 8) | (b << 16);
#endif

	// write color to the buffer
	const uint32_t fbIndex = ix + iy * optixLaunchParams.resolutionX;
	optixLaunchParams.colorBuffer[fbIndex] = rgba;
}
