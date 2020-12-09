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

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const int32_t pathIx = __float_as_int(O4.w);
	const int32_t onSensor = __float_as_int(D4.w);
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	// didn't hit anything
	if(primIx == ~0)
	{
		accumulator[pixelIx] += make_float4(onSensor, onSensor, onSensor, 0);
		return;
	}

	const float z = clamp(tmax * dot(D, params->cameraForward) / params->zDepthMax, 0.f, 1.f);
	accumulator[pixelIx] += make_float4(z, z, z, 0);
}
