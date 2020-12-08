#pragma once

#include "CudaUtility.h"

__global__ void ZDepthKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIdx >= pathCount)
		return;

	// gather data
	const float4 O4 = pathStates[jobIdx + (stride * 0)];
	const float4 D4 = pathStates[jobIdx + (stride * 1)];
	const float4 T4 = pathLength == 0 ? make_float4(1) : pathStates[jobIdx + (stride * 2)];

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const float3 T = make_float3(T4);
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	// didn't hit anything
	if(primIx == ~0)
	{
		accumulator[pixelIx] += make_float4(T * SampleSky(D, skyData->drawSun));
		return;
	}

	const float z = tmax * dot(D, params->cameraForward) / params->zDepthMax;
	accumulator[pixelIx] += make_float4(z, z, z, 0);
}
