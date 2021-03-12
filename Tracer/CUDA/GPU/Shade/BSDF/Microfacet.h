// Adapted from AppleSeed
// https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/foundation/math/microfacet.cpp
// https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/microfacethelper.h

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

#include "CudaLinearMath.h"

namespace Microfacet
{
	//--------------------------------------------------------------------------------------------------------------------------
	// Microfacet alpha
	//--------------------------------------------------------------------------------------------------------------------------
	static inline __device__
	float alpha_from_roughness(const float roughness)
	{
		return max(0.001f, roughness * roughness);
	}



	static inline __device__
	void alpha_from_roughness(const float roughness, const float anisotropy, float& alpha_x, float& alpha_y)
	{
		const float squareRoughness = square(roughness);
		//const float aspect = sqrtf(1.0f - anisotropy * copysignf(0.9f, anisotropy));
		const float aspect = sqrtf(1.0f + anisotropy * (anisotropy < 0 ? 0.9f : -0.9f));
		alpha_x = max(0.001f, squareRoughness / aspect);
		alpha_y = max(0.001f, squareRoughness * aspect);
	}



	static inline __device__
	void sample_phi(float s, float& cos_phi, float& sin_phi)
	{
		const float phi = TwoPi * s;
		cos_phi = cosf(phi);
		sin_phi = sinf(phi);
	}



	static inline __device__
	float stretched_roughness(const float3& m, float sin_theta, float alpha_x, float alpha_y)
	{
		if (alpha_x == alpha_y || sin_theta == 0.0f)
			return 1.0f / square(alpha_x);

		const float cos_phi_2_ax_2 = square(m.x / (sin_theta * alpha_x));
		const float sin_phi_2_ay_2 = square(m.z / (sin_theta * alpha_y));
		return cos_phi_2_ax_2 + sin_phi_2_ay_2;
	}



	static inline __device__
	float projected_roughness(const float3& m, const float sin_theta, const float alpha_x, const float alpha_y)
	{
		if (alpha_x == alpha_y || sin_theta == 0.0f)
			return alpha_x;

		const float cos_phi_2_ax_2 = square((m.x * alpha_x) / sin_theta);
		const float sin_phi_2_ay_2 = square((m.z * alpha_y) / sin_theta);
		return sqrtf(cos_phi_2_ax_2 + sin_phi_2_ay_2);
	}



	static inline __device__
	float3 make_unit_vector(float cos_theta, float sin_theta, float cos_phi, float sin_phi)
	{
		// https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/foundation/math/vector.h
		return make_float3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
	}



	namespace GGX
	{
		//
		// Anisotropic GGX Microfacet Distribution Function.
		//
		// References:
		//
		//   [1] Microfacet Models for Refraction through Rough Surfaces
		//       http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
		//
		//   [2] Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs
		//       http://hal.inria.fr/docs/00/96/78/44/PDF/RR-8468.pdf
		//
		//   [3] Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals.
		//       https://hal.inria.fr/hal-00996995/en
		//
		//   [4] A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals.
		//       https://hal.archives-ouvertes.fr/hal-01509746
		//

		static inline __device__
		float Lambda(const float3& v, float alpha_x, float alpha_y)
		{
			const float cos_theta = v.y;
			if (cos_theta == 0.0f)
				return 0.0f;

			const float cos_theta_2 = square(cos_theta);
			const float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta_2));

			const float alpha = projected_roughness(v, sin_theta, alpha_x, alpha_y);

			const float tan_theta_2 = square(sin_theta) / cos_theta_2;
			const float a2_rcp = square(alpha) * tan_theta_2;
			return (-1.0f + sqrtf(1.0f + a2_rcp)) * 0.5f;
		}



		static inline __device__
		float D(const float3& m, float alpha_x, float alpha_y)
		{
			const float cos_theta = m.z;
			if (cos_theta == 0.0f)
				return square(alpha_x) * RcpPi;

			const float cos_theta_2 = square(cos_theta);
			const float sin_theta   = sqrtf(max(0.0f, 1.0f - cos_theta_2));
			const float cos_theta_4 = square(cos_theta_2);
			const float tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2;

			const float A = stretched_roughness(m, sin_theta, alpha_x, alpha_y);

			const float tmp = 1.0f + tan_theta_2 * A;
			return 1.0f / (Pi * alpha_x * alpha_y * cos_theta_4 * square(tmp));
		}



		static inline __device__
		float G(const float3& wi, const float3& wo, const float3& m, float alpha_x, float alpha_y)
		{
			return 1.0f / (1.0f + Lambda(wo, alpha_x, alpha_y) + Lambda(wi, alpha_x, alpha_y));
		}



		static inline __device__
		float G1(const float3& v, const float3& m, float alpha_x, float alpha_y)
		{
			return 1.0f / (1.0f + Lambda(v, alpha_x, alpha_y));
		}



		static inline __device__
		float3 Sample(const float3& v, float r0, float r1, float alpha_x, float alpha_y)
		{
			// Stretch incident.
			//const float sign_cos_vn = copysignf(1.0f, v.z);
			const float sign_cos_vn = v.z < 0.0f ? -1.0f : 1.0f;
			float3 stretched = make_float3(sign_cos_vn * v.x * alpha_x, sign_cos_vn * v.y * alpha_y, sign_cos_vn * v.z);
			stretched = normalize(stretched);

			// Build an orthonormal basis.
			const float3 t1 =
				v.y < 0.9999f
					? normalize(cross(stretched, make_float3(0.0f, 0.0f, 1.0f)))
					: make_float3(1.0f, 0.0f, 0.0f);
			const float3 t2 = cross(t1, stretched);

			// Sample point with polar coordinates (r, phi).
			const float a = 1.0f / (1.0f + stretched.z);
			const float r = sqrtf(r0);
			const float phi =
				r1 < a
					? r1 / a * Pi
					: Pi + (r1 - a) / (1.0f - a) * Pi;

			const float p1 = r * cosf(phi);
			const float p2 = r * sinf(phi) * (r1 < a ? 1.0f : stretched.z);

			// Compute normal.
			const float3 h = p1 * t1 + p2 * t2 + sqrtf(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * stretched;

			// Unstretch and normalize.
			const float3 m = make_float3(
				h.x * alpha_x,
				h.y * alpha_y,
				max(0.0f, h.z));
			return normalize(m);
		}



		static inline __device__
		float Pdf(const float3& v, const float3& m, float alpha_x, float alpha_y)
		{
			// return pdf_visible_normals(v, m, alpha_x, alpha_y);
			const float cos_theta_v = v.z;

			if (cos_theta_v == 0.0f)
				return 0.0f;

			return
				G1(v, m, alpha_x, alpha_y) * fabsf(dot(v, m)) *
				D(m, alpha_x, alpha_y) / fabsf(cos_theta_v);
		}
	}



	namespace GTR1
	{
		//
		// GTR1 Microfacet Distribution Function.
		//
		// References:
		//
		//   [1] Physically-Based Shading at Disney
		//       https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
		//
		//   [2] Deriving the Smith shadowing function G1 for gamma (0, 4]
		//       https://docs.chaosgroup.com/download/attachments/7147732/gtr_shadowing.pdf?version=2&modificationDate=1434539612000&api=v2
		//

		static inline __device__
		float Lambda(const float3& v, float alpha_x, float alpha_y)
		{
			const float cos_theta = v.z;
			if (cos_theta == 0.0f)
				return 0.0f;

			// [2] section 3.2.
			const float cos_theta_2 = square(cos_theta);
			const float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta_2));

			// Normal incidence. No shadowing.
			if (sin_theta == 0.0f)
				return 0.0f;

			const float cot_theta_2 = cos_theta_2 / square(sin_theta);
			const float cot_theta = sqrtf(cot_theta_2);
			const float alpha = clamp(alpha_x, 0.001f, 0.999f);
			const float alpha_2 = square(alpha);

			const float a = sqrtf(cot_theta_2 + alpha_2);
			const float b = sqrtf(cot_theta_2 + 1.0f);
			const float c = logf(cot_theta + b);
			const float d = logf(cot_theta + a);

			return (a - b + cot_theta * (c - d)) / (cot_theta * logf(alpha_2));
		}



		static inline __device__
		float D(const float3& m, float alpha_x, float alpha_y)
		{
			 const float alpha = clamp(alpha_x, 0.001f, 0.999f);
			const float alpha_x_2 = square(alpha);
			const float cos_theta_2 = square(m.z);
			const float a = (alpha_x_2 - 1.0f) / (Pi * logf(alpha_x_2));
			const float b = (1.0f / (1.0f + (alpha_x_2 - 1.0f) * cos_theta_2));
			return a * b;
		}



		static inline __device__
		float G(const float3& wi, const float3& wo, const float3& m, float alpha_x, float alpha_y)
		{
			return 1.0f / (1.0f + Lambda(wo, alpha_x, alpha_y) + Lambda(wi, alpha_x, alpha_y));
		}



		static inline __device__
		float G1(const float3& v, const float3& m, float alpha_x, float alpha_y)
		{
			return 1.0f / (1.0f + Lambda(v, alpha_x, alpha_y));
		}



		static inline __device__
		float3 Sample(const float3& v, float r0, float r1, float alpha_x, float alpha_y)
		{
			const float alpha = clamp(alpha_x, 0.001f, 0.999f);
			const float alpha_2 = square(alpha);
			const float a = 1.0f - powf(alpha_2, 1.0f - r0);
			const float cos_theta_2 = a / (1.0f - alpha_2);
			const float cos_theta = sqrtf(cos_theta_2);
			const float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta_2));

			float cos_phi, sin_phi;
			sample_phi(r1, cos_phi, sin_phi);
			return make_unit_vector(cos_theta, sin_theta, cos_phi, sin_phi);
		}


		static inline __device__
		float Pdf(const float3& v, const float3& m, float alpha_x, float alpha_y)
		{
			return D(m, alpha_x, alpha_y) * fabsf(m.y);
		}
	}
}
