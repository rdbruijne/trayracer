#pragma once

#include "CudaUtility.h"

__global__ void ObjectIDKernel(uint32_t pathCount, float4* accumulator, float4* pathStates, uint4* hitData, float4* shadowRays, int2 resolution, uint32_t stride, uint32_t pathLength)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	// gather data
	const uint4 hd = hitData[pathIx];
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	// ID to color
	accumulator[pixelIx] += make_float4(IdToColor(instIx + 1), 0);
}
