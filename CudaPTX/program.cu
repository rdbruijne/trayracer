// OptiX
#include <optix_device.h>

// Project
#include "CommonStructs.h"

/*!
 * globals
 */
extern "C" __constant__ LaunchParams optixLaunchParams;



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
}



/*!
 * @brief Miss program.
 */
extern "C" __global__ void __miss__radiance()
{
}



/*!
 * @brief Ray gen program.
 */
extern "C" __global__ void __raygen__renderFrame()
{
	// get the current pixel index
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	// generate a color
	const int r = (ix + optixLaunchParams.frameID) & 255;
	const int g = (iy + optixLaunchParams.frameID * 2) & 255;
	const int b = (ix + iy) & 255;
	const uint32_t rgba = 0xFF000000 | (r << 0) | (g << 8) | (b << 16);

	// write color to the buffer
	const uint32_t fbIndex = ix + iy * optixLaunchParams.resolution.x;
	optixLaunchParams.colorBuffer[fbIndex] = rgba;
}
