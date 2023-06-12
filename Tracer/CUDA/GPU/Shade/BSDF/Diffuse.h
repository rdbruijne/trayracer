#pragma once

#include "Closure.h"
#include "CudaUtility.h"

namespace Diffuse
{
	static inline __device__
	BsdfResult Evaluate(const float3& wo, const float3& wi, const HitMaterial& mat)
	{
		BsdfResult result;
		result.wi  = wi;
		result.pdf = RcpPi * wi.z;
		result.T   = make_float3(RcpPi) * mat.diffuse;
		return result;
	}



	static inline __device__
	BsdfResult Sample(const float3& wo, const HitMaterial& mat, float r0, float r1)
	{
		const float3 wi = SampleCosineHemisphere(r0, r1);

		BsdfResult result;
		result.wi    = wi;
		result.pdf   = RcpPi * wi.z;
		result.T     = make_float3(RcpPi) * mat.diffuse;
		result.flags = BsdfFlags::None;
		return result;
	}
}



static inline __device__
Closure DiffuseClosure(const ShadingInfo& shadingInfo, const HitMaterial& mat, float r0, float r1)
{
	const BsdfResult eval   = Diffuse::Evaluate(shadingInfo.wo, shadingInfo.wi, mat);
	const BsdfResult sample = Diffuse::Sample(shadingInfo.wo, mat, r0, r1);
	return FinalizeClosure(shadingInfo, eval, sample);
}
