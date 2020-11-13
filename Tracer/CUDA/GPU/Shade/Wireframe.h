#pragma once

#include "CudaUtility.h"

__global__ void WireframeKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];

	const float3 O = make_float3(O4);
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	const uint4 hd = hitData[pathIx];
	const uint32_t primIx = hd.z;

	if(pathLength == 0)
	{
		float3 newOrigin, newDir;
		uint32_t seed = tea<2>(pathIx, params->sampleCount + pathLength + 1);
		GenerateCameraRay(newOrigin, newDir, make_int2(pixelIx % params->resX, pixelIx / params->resX), seed);

		// update path states
		const int32_t extendIx = atomicAdd(&counters->extendRays, 1);
		pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __int_as_float(pathIx));
		pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);
		pathStates[extendIx + (stride * 2)] = make_float4(__uint_as_float(primIx));
	}
	else
	{
		const float4 T4 = pathStates[jobIdx + (stride * 2)];
		const uint32_t prevT = __float_as_uint(T4.w);
		if(prevT == primIx)
			accumulator[pixelIx] += make_float4(1, 1, 1, 0);
	}
}
