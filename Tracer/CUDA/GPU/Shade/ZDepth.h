#pragma once

__global__ __launch_bounds__(128, 2)
void ZDepthKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIx + (stride * 0)];
	const float4 D4 = pathStates[jobIx + (stride * 1)];

	// extract path data
	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);
	const uint32_t pathIx = PathIx(__float_as_uint(O4.w));
	const uint32_t pixelIx = pathIx % (resolution.x * resolution.y);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	// didn't hit anything
	if(primIx == ~0)
		return;

	const float z = clamp(tmax * dot(D, Params->cameraForward) / Params->kernelSettings.zDepthMax, 0.f, 1.f);
	accumulator[pixelIx] += make_float4(z, z, z, 0);
}
