#pragma once

// CUDA math classes
#pragma warning(push)
#pragma warning(disable: 4244)  // 'argument' : conversion from 'type1' to 'type2', possible loss of data
#pragma warning(disable: 4365)  // 'argument': conversion from '%1' to '%2', signed/unsigned mismatch
#pragma warning(disable: 28251) // Inconsistent annotation for '%1': this instance has no annotations.
#include "CUDA/helper_math.h"
#pragma warning(pop)

namespace Tracer
{
	//
	// extra constants
	//
	constexpr float Epsilon  = 1e-3f;
	constexpr float DegToRad = 3.14159265359f / 180.f;
	constexpr float RadToDeg = 180.f / 3.14159265359f;



	//
	// extension functions
	//
	inline float2 operator -(const float2& a) { return make_float2(-a.x, -a.y); }
	inline float3 operator -(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
	inline float4 operator -(const float4& a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }



	//
	// vector math
	//
	static float3 RotateAroundAxis(float3 v, float3 axis, float radians)
	{
		// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
		const float cosAngle = cosf(radians);
		const float sinAngle = sinf(radians);
		return (v * cosAngle) + (cross(axis, v) * sinAngle) + (axis * dot(axis, v) * (1.f - cosAngle));
	}



	static float3 SmallestAxis(const float3& v)
	{
		// squaring is faster than abs
		const float x2 = v.x * v.x;
		const float y2 = v.y * v.y;
		const float z2 = v.z * v.z;

		if(x2 < y2)
			return x2 < z2 ? make_float3(1, 0, 0) : make_float3(0, 0, 1);
		return y2 < z2 ? make_float3(0, 1, 0) : make_float3(0, 0, 1);
	}
}
