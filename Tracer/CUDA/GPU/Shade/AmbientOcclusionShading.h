#pragma once

__global__ __launch_bounds__(128, 2)
void AmbientOcclusionShadingKernel(DECLARE_KERNEL_PARAMS)
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
	const uint32_t pixelIx = pathIx % (resX * resY);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;
	const float tmax = __uint_as_float(hd.w);

	if(pathLength == 0)
	{
		// didn't hit anything
		if(primIx == ~0)
			return;

		// fetch intersection info
		Intersection intersection = {};
		HitMaterial hitMaterial = {};
		GetIntersectionAttributes(instIx, primIx, bary, intersection, hitMaterial);

		// diffuse
		float3 diff = hitMaterial.diffuse;

		// bounce ray
		uint32_t seed = tea<2>(Params->kernelSettings.seed + pathIx, Params->sampleCount + pathLength + 1);
		const float3 newOrigin = O + (D * tmax);
		const float3 newDir = SampleCosineHemisphere(intersection.shadingNormal, rnd(seed), rnd(seed));

		// update path states
		__threadfence();
		const int32_t extendIx = atomicAdd(&Counters->extendRays, 1);
		__threadfence();
		pathStates[extendIx + (stride * 0)] = make_float4(newOrigin, __uint_as_float(Pack(pathIx)));
		pathStates[extendIx + (stride * 1)] = make_float4(newDir, 0);
		pathStates[extendIx + (stride * 2)] = make_float4(diff, 0);

		// denoiser data
		albedo[pixelIx] = make_float4(hitMaterial.diffuse, 0);
		normals[pixelIx] = make_float4(intersection.shadingNormal, 0);
	}
	else
	{
		const float4 T4 = pathStates[jobIx + (stride * 2)];
		const float3 T = make_float3(T4);

		const float z = (tmax > Params->kernelSettings.aoDist) ? 1.f : tmax / Params->kernelSettings.aoDist;
		accumulator[pixelIx] += make_float4(T * z, 0);
	}
}
