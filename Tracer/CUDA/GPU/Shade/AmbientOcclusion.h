#pragma once

#include "CudaUtility.h"

__global__ void AmbientOcclusionKernel(DECLARE_KERNEL_PARAMS)
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
	const int32_t pixelIx = pathIx % (resolution.x * resolution.y);

	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	if(pathLength == 0)
	{
		if(primIx == ~0)
			return;
		uint32_t seed = tea<2>(pathIx, params->sampleCount + pathLength + 1);

		// fetch intersection info
		const IntersectionAttributes attrib = GetIntersectionAttributes(instIx, primIx, bary);

		// fix infacing normal
		const float3 newOrigin = O + (D * tmax);
		const float3 newDir = SampleCosineHemisphere(attrib.shadingNormal, rnd(seed), rnd(seed));

		// update path states
		const int32_t extendIx = atomicAdd(&counters->extendRays, 1);
		pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __int_as_float(pathIx));
		pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);

		// denoiser data
		albedo[pixelIx] = make_float4(1, 1, 1, 0);
		normals[pixelIx] = make_float4(attrib.shadingNormal, 0);
	}
	else
	{
		const float z = (tmax > params->aoDist) ? 1.f : tmax / params->aoDist;
		accumulator[pixelIx] += make_float4(z, z, z, 0);
	}
}
