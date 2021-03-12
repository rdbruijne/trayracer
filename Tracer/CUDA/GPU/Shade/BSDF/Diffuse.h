#pragma once

#include "Closure.h"
#include "CudaUtility.h"

namespace Diffuse
{
	static inline __device__
	BsdfResult Evaluate(const float3& wo, const float3& wi)
	{
		const float pdf = 1.f / Pi;

		BsdfResult result;
		result.wi  = wi;
		result.pdf = pdf;
		result.T   = make_float3(pdf * wi.z);
		return result;
	}



	static inline __device__
	BsdfResult Sample(const float3& wo, float r0, float r1)
	{
		return Evaluate(wo, SampleHemisphere(r0, r1));
	}
}



static inline __device__
Closure DiffuseClosure(const ShadingInfo& shadingInfo, const HitMaterial& /*mat*/, float r0, float r1)
{
	const BsdfResult eval   = Diffuse::Evaluate(shadingInfo.wo, shadingInfo.wi);
	const BsdfResult sample = Diffuse::Sample(shadingInfo.wo, r0, r1);
	return FinalizeClosure(shadingInfo, eval, sample);
}
