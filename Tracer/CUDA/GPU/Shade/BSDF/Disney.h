// Adapted from AppleSeed
// https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/disneybrdf.cpp

//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2014-2018 Esteban Tovagliari, The appleseedhq Organization
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//

#pragma once

#include "Closure.h"
#include "BSDF/Microfacet.h"

namespace Disney
{
	static inline __device__
	float3 SurfaceNormal()
	{
		return make_float3(0.f, 0.f, 1.f);
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Spectra
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	float3 mix_spectra(const float3& a, const float3& b, float t)
	{
		return ((1.f - t) * a) + (t * b);
	}



	static inline __device__
	float3 mix_one_with_spectra(const float3& b, float t)
	{
		return (1.f - t) + (t * b);
	}



	static inline __device__
	float3 mix_spectra_with_one(const float3& a, float t)
	{
		return ((1.f - t) * a) + t;
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Clearcoat roughness
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	float clearcoat_roughness(const HitMaterial& mat)
	{
		return mix(0.1f, 0.001f, mat.clearcoatGloss);
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Fresnel
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	float schlick_fresnel(float u)
	{
		const float m = __saturatef(1.0f - u);
		const float m2 = square(m);
		const float m4 = square(m2);
		return m4 * m;
	}



	static inline __device__
	float3 DisneySpecularFresnel(const HitMaterial& mat, const float3& o, const float3& h)
	{
		float3 value = mix_one_with_spectra(mat.tint, mat.specularTint);
		value *= mat.specular * 0.08f;
		value = mix_spectra(value, mat.diffuse, mat.metallic);
		const float cos_oh = fabsf(dot(o, h));
		return mix_spectra_with_one(value, schlick_fresnel(cos_oh));
	}



	static inline __device__
	float3 DisneyClearcoatFresnel(const HitMaterial& mat, const float3& o, const float3& h)
	{
		const float cos_oh = fabsf(dot(o, h));
		return make_float3(mix(0.04f, 1.0f, schlick_fresnel(cos_oh)) * 0.25f * mat.clearcoat);
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Force above surface
	// https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/bsdf.h
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	bool force_above_surface(float3& direction, const float3& normal)
	{
		const float Eps = 1.0e-4f;

		const float cos_theta = dot(direction, normal);
		const float correction = Eps - cos_theta;

		if(correction <= 0)
			return false;

		direction = normalize(direction + correction * normal);
		return true;
	}



	static inline __device__
	bool force_above_surface(float3& direction)
	{
		return force_above_surface(direction, SurfaceNormal());
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Diffuse
	//--------------------------------------------------------------------------------------------------------------------------
	namespace Diffuse
	{
		static inline __device__
		float Evaluate(const HitMaterial& mat, const float3& wo, const float3& wi,
							  float3& value)
		{
			// This code is mostly ported from the GLSL implementation in Disney's BRDF explorer.

			const float3 n = SurfaceNormal();
			const float3 h = normalize(wi + wo);

			// Using the absolute values of cos_on and cos_in creates discontinuities.
			const float cos_on = dot(n, wo);
			const float cos_in = dot(n, wi);
			const float cos_ih = dot(wi, h);

			const float fl = schlick_fresnel(cos_in);
			const float fv = schlick_fresnel(cos_on);
			float fd = 0.0f;

			if(mat.subsurface != 1.0f)
			{
				const float fd90 = 0.5f + 2.0f * square(cos_ih) * mat.roughness;
				fd = mix(1.0f, fd90, fl) * mix(1.0f, fd90, fv);
			}

			if (mat.subsurface > 0.0f)
			{
				// Based on Hanrahan-Krueger BRDF approximation of isotropic BSRDF.
				// The 1.25 scale is used to (roughly) preserve albedo.
				// Fss90 is used to "flatten" retroreflection based on roughness.
				const float fss90 = square(cos_ih) * mat.roughness;
				const float fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);
				const float ss = 1.25f * (fss * (1.0f / (fabsf(cos_on) + fabsf(cos_in)) - 0.5f) + 0.5f);
				fd = mix(fd, ss, mat.subsurface);
			}

			value = mat.diffuse;
			value *= fd * RcpPi * (1.0f - mat.metallic);

			// Return the probability density of the sampled direction.
			return fabsf(cos_in) * RcpPi;
		}



		static inline __device__
		void Sample(const HitMaterial& mat, float r0, float r1, const float3& wo,
						   float3& wi, float& pdf, float3& value)
		{
			// Compute the incoming direction.
			wi = SampleCosineHemisphere(r0, r1);

			// Compute the component value and the probability density of the sampled direction.
			pdf = Evaluate(mat, wo, wi, value);
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Sheen
	//--------------------------------------------------------------------------------------------------------------------------
	namespace Sheen
	{
		static inline __device__
		float Evaluate(const HitMaterial& mat, const float3& wo, const float3& wi,
							  float3& value)
		{
			// This code is mostly ported from the GLSL implementation in Disney's BRDF explorer.

			// Compute the component value.
			const float3 h = normalize(wi + wo);
			const float cos_ih = dot(wi, h);
			const float fh = schlick_fresnel(cos_ih);
			value = mix_one_with_spectra(mat.tint, mat.sheenTint);
			value *= fh * mat.sheen * (1.0f - mat.metallic);

			// Return the probability density of the sampled direction.
			return RcpTwoPi;
		}



		static inline __device__
		void Sample(const HitMaterial& mat, float r0, float r1, const float3& wo,
						   float3& wi, float& pdf, float3& value)
		{
			// Compute the incoming direction.
			wi = SampleCosineHemisphere(r0, r1);

			// Compute the component value and the probability density of the sampled direction.
			pdf = Evaluate(mat, wo, wi, value);
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Microfacet GGX
	//--------------------------------------------------------------------------------------------------------------------------
	namespace GGX
	{
		static inline __device__
		float Evaluate(float alpha_x, float alpha_y, const HitMaterial& mat, const float3& wo, const float3& wi,
							  float3& value)
		{
			if(wo.z == 0.0f || wi.z == 0.0f)
				return 0.0f;

			const float3 m = normalize(wi + wo);
			const float cos_oh = dot(wo, m);
			if(cos_oh == 0.0f)
				return 0.0f;

			const float D = Microfacet::GGX::D(m, alpha_x, alpha_y);
			const float G = Microfacet::GGX::G(wi, wo, m, alpha_x, alpha_y);

			value = DisneySpecularFresnel(mat, wo, m);

			const float cos_on = wo.z;
			const float cos_in = wi.z;
			value *= D * G / fabsf(4.0f * cos_on * cos_in);

			return Microfacet::GGX::Pdf(wo, m, alpha_x, alpha_y) / fabsf(4.0f * cos_oh);
		}



		static inline __device__
		void Sample(float alpha_x, float alpha_y, const HitMaterial& mat, float r0, float r1, const float3& wo,
						   float3& wi, float& pdf, float3& value)
		{
			if(wo.z == 0.0f)
				return;

			// Compute the incoming direction by sampling the MDF.
			float3 m = Microfacet::GGX::Sample(wo, r0, r1, alpha_x, alpha_y);
			wi = reflect(-wo, m);

			// Force the outgoing direction to lie above the geometric surface.
			if(force_above_surface(wi))
				m = normalize(wo + wi);

			if (wi.z == 0.0f)
				return;

			const float cos_oh = dot(wo, m);
			pdf = Microfacet::GGX::Pdf(wo, m, alpha_x, alpha_y) / fabsf(4.0f * cos_oh);

			// Skip samples with very low probability.
			if(pdf < 1e-6f)
				return;

			const float D = Microfacet::GGX::D(m, alpha_x, alpha_y);
			const float G = Microfacet::GGX::G(wi, wo, m, alpha_x, alpha_y);
			value = DisneySpecularFresnel(mat, wo, m);
			value *= D * G / fabs(4.0f * wo.z * wi.z);
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Microfacet GTR1
	//--------------------------------------------------------------------------------------------------------------------------
	namespace GTR1
	{
		static inline __device__
		float Evaluate(float alpha_x, float alpha_y, const HitMaterial& mat, const float3& wo, const float3& wi,
							  float3& value)
		{
			if(wo.z == 0.0f || wi.z == 0.0f)
				return 0.0f;

			const float3 m = normalize(wi + wo);
			const float cos_oh = dot(wo, m);
			if(cos_oh == 0.0f)
				return 0.0f;

			const float D = Microfacet::GTR1::D(m, alpha_x, alpha_y);
			const float G = Microfacet::GTR1::G(wi, wo, m, alpha_x, alpha_y);

			value = DisneyClearcoatFresnel(mat, wo, m);

			const float cos_on = wo.z;
			const float cos_in = wi.z;
			value *= D * G / fabsf(4.0f * cos_on * cos_in);

			return Microfacet::GTR1::Pdf(wo, m, alpha_x, alpha_y) / fabsf(4.0f * cos_oh);
		}



		static inline __device__
		void Sample(float alpha_x, float alpha_y, const HitMaterial& mat, float r0, float r1, const float3& wo,
						   float3& wi, float& pdf, float3& value)
		{
			if(wo.z == 0.0f)
				return;

			// Compute the incoming direction by sampling the MDF.
			float3 m = Microfacet::GTR1::Sample(wo, r0, r1, alpha_x, alpha_y);
			wi = reflect(-wo, m);

			// Force the outgoing direction to lie above the geometric surface.
			if(force_above_surface(wi))
				m = normalize(wo + wi);

			if (wi.z == 0.0f)
				return;

			const float cos_oh = dot(wo, m);
			pdf = Microfacet::GTR1::Pdf(wo, m, alpha_x, alpha_y) / fabsf(4.0f * cos_oh);

			// Skip samples with very low probability.
			if(pdf < 1e-6f)
				return;

			const float D = Microfacet::GTR1::D(m, alpha_x, alpha_y);
			const float G = Microfacet::GTR1::G(wi, wo, m, alpha_x, alpha_y);
			value = DisneySpecularFresnel(mat, wo, m);
			value *= D * G / fabs(4.0f * wo.z * wi.z);
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Weights
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	bool compute_component_weights(const HitMaterial& mat,
		float& diffuseWeight, float& sheenWeight, float& specularWeight, float& clearcoatWeight)
	{
		// Compute component weights.
		diffuseWeight   = lerp(mat.luminance, 0.f, mat.metallic);
		sheenWeight     = lerp(mat.sheen,     0.f, mat.metallic);
		specularWeight  = lerp(mat.specular,  1.f, mat.metallic);
		clearcoatWeight = mat.clearcoat * .25f;

		const float total_weight = diffuseWeight + sheenWeight + specularWeight + clearcoatWeight;
		if (total_weight == 0.0f)
			return false;

		const float rcp_total_weight = 1.f / total_weight;
		diffuseWeight   *= rcp_total_weight;
		sheenWeight     *= rcp_total_weight;
		specularWeight  *= rcp_total_weight;
		clearcoatWeight *= rcp_total_weight;

		return true;
	}


	//--------------------------------------------------------------------------------------------------------------------------
	// Evaluate BSDF
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	BsdfResult Evaluate(const ShadingInfo& shadingInfo, const HitMaterial& mat)
	{
		// Compute component weights.
		float diffuseWeight;
		float sheenWeight;
		float specularWeight;
		float clearcoatWeight;
		if(!compute_component_weights(mat, diffuseWeight, sheenWeight, specularWeight, clearcoatWeight))
			return BsdfResult{};

		// Compute pdf
		float pdf = 0;
		float3 incoming = make_float3(0);

		// Diffuse contribution
		if(diffuseWeight > 0)
			pdf += diffuseWeight * Diffuse::Evaluate(mat, shadingInfo.wo, shadingInfo.wi, incoming);

		// Sheen contribution
		if(sheenWeight > 0)
			pdf += sheenWeight * Sheen::Evaluate(mat, shadingInfo.wo, shadingInfo.wi, incoming);

		// Specular contribution
		if(specularWeight > 0)
		{
			float alpha_x, alpha_y;
			Microfacet::alpha_from_roughness(mat.roughness, mat.anisotropic, alpha_x, alpha_y);

			float3 specularValue;
			const float specularPdf = GGX::Evaluate(alpha_x, alpha_y, mat, shadingInfo.wo, shadingInfo.wi, specularValue);
			if(specularPdf > 0)
			{
				pdf += specularWeight * specularPdf;
				incoming += specularValue;
			}
		}

		// Clearcoat contribution
		if(clearcoatWeight > 0)
		{
			const float alpha = clearcoat_roughness(mat);

			float3 clearcoatValue;
			const float clearcoatPdf = GTR1::Evaluate(alpha, alpha, mat, shadingInfo.wo, shadingInfo.wi, clearcoatValue);
			if(clearcoatPdf > 0)
			{
				pdf += clearcoatWeight * clearcoatPdf;
				incoming += clearcoatValue;
			}
		}

		// Create closure
		BsdfResult result;
		result.wi  = shadingInfo.wi;
		result.pdf = pdf;
		result.T   = incoming;
		return result;
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Sample BSDF
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	BsdfResult Sample(const ShadingInfo& shadingInfo, const HitMaterial& mat, float r0, float r1)
	{
		// Compute component weights.
		float diffuseWeight;
		float sheenWeight;
		float specularWeight;
		float clearcoatWeight;
		if(!compute_component_weights(mat, diffuseWeight, sheenWeight, specularWeight, clearcoatWeight))
			return BsdfResult{};

		// Compute CDF to sample components.
		const float diffuseCdf   = diffuseWeight;
		const float sheenCdf     = diffuseCdf + sheenWeight;
		const float specularCdf  = sheenCdf + specularWeight;
		//const float clearcoatCdf = specularCdf + clearcoatWeight;

		// Choose which of the components to sample.
		float3 wi;
		float probability;
		float component_pdf;
		float3 value = make_float3(0);

		// Sample the chosen component.
		if(r0 < diffuseCdf)
		{
			Diffuse::Sample(mat, r0, r1, shadingInfo.wo, wi, component_pdf, value);
			probability = diffuseWeight * component_pdf;
			diffuseWeight = 0;
		}
		else if(r0 < sheenCdf)
		{
			Sheen::Sample(mat, r0, r1, shadingInfo.wo, wi, component_pdf, value);
			probability = sheenWeight * component_pdf;
			sheenWeight = 0;
		}
		else if(r0 < specularCdf)
		{
			float alpha_x;
			float alpha_y;
			Microfacet::alpha_from_roughness(mat.roughness, mat.anisotropic, alpha_x, alpha_y);
			GGX::Sample(alpha_x, alpha_y, mat, r0, r1, shadingInfo.wo, wi, component_pdf, value);
			probability = specularWeight * component_pdf;
			specularWeight = 0;
		}
		else
		{
			const float alpha = clearcoat_roughness(mat);
			GTR1::Sample(alpha, alpha, mat, r0, r1, shadingInfo.wo, wi, component_pdf, value);
			probability = clearcoatWeight * component_pdf;
			clearcoatWeight = 0;
		}

		// Evaluate the components.
		if(diffuseWeight > 0)
		{
			float3 diffuseValue;
			probability += diffuseWeight * Diffuse::Evaluate(mat, shadingInfo.wo, wi, diffuseValue);
			value += diffuseValue;
		}

		if(sheenWeight > 0)
		{
			float3 sheenValue;
			probability += sheenWeight * Sheen::Evaluate(mat, shadingInfo.wo, wi, sheenValue);
			value += sheenValue;
		}

		if(specularWeight > 0)
		{
			float alpha_x;
			float alpha_y;
			Microfacet::alpha_from_roughness(mat.roughness, mat.anisotropic, alpha_x, alpha_y);

			float3 specularValue;
			probability += specularWeight * GGX::Evaluate(alpha_x, alpha_y, mat, shadingInfo.wo, wi, specularValue);
			value += specularValue;
		}

		if(clearcoatWeight > 0)
		{
			const float alpha = clearcoat_roughness(mat);

			float3 clearcoatValue;
			GTR1::Evaluate(alpha, alpha, mat, shadingInfo.wo, wi, clearcoatValue);
			value += clearcoatValue;
		}

		// Create closure
		BsdfResult result;
		result.wi    = wi;
		result.pdf   = probability > 1e-6f ? probability : 0;
		result.T     = value;
		result.flags = mat.roughness < Epsilon ? BsdfFlags::Specular : BsdfFlags::None;
		return result;
	}
}



//------------------------------------------------------------------------------------------------------------------------------
// Closure
//------------------------------------------------------------------------------------------------------------------------------
static inline __device__
Closure DisneyClosure(const ShadingInfo& shadingInfo, const HitMaterial& mat, float r0, float r1)
{
	const BsdfResult eval   = Disney::Evaluate(shadingInfo, mat);
	const BsdfResult sample = Disney::Sample(shadingInfo, mat, r0, r1);
	return FinalizeClosure(shadingInfo, eval, sample);
}
