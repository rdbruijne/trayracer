#pragma once
#pragma once

#include "Closure.h"
#include "CudaUtility.h"

namespace Lambert
{
	static inline __device__
	BsdfResult Evaluate(const float3& wo, const float3& wi, const HitMaterial& mat)
	{
		const float pdf = mat.roughness < Epsilon ? 0 : RcpPi;

		BsdfResult result;
		result.wi  = wi;
		result.pdf = pdf * wi.z;
		result.T   = make_float3(pdf);
		return result;
	}



	static inline __device__
	BsdfResult Sample(const float3& wo, const HitMaterial& mat, float r0, float r1)
	{
		BsdfResult result;
		const float specProb = 1.f - mat.roughness;
		if(r0 < specProb)
		{
			// pure specular
			const float3 wi = reflect(-wo, make_float3(0, 0, 1));
			result.wi    = wi;
			result.pdf   = 1;
			result.T     = make_float3(1.f / wi.z) * mat.diffuse;
			result.flags = BsdfFlags::Specular;
			return result;
		}
		else
		{
			// diffuse
			const float3 wi = SampleCosineHemisphere((specProb - r0) / (1.f - specProb), r1);

			result.wi    = wi;
			result.pdf   = RcpPi * wi.z;
			result.T     = make_float3(RcpPi) * mat.diffuse;
			result.flags = BsdfFlags::None;
		}
		return result;
	}
}



static inline __device__
Closure LambertClosure(const ShadingInfo& shadingInfo, const HitMaterial& mat, float r0, float r1)
{
	const BsdfResult eval   = Lambert::Evaluate(shadingInfo.wo, shadingInfo.wi, mat);
	const BsdfResult sample = Lambert::Sample(shadingInfo.wo, mat, r0, r1);
	return FinalizeClosure(shadingInfo, eval, sample);
}
