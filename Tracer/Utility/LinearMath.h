#pragma once

// CUDA math classes
#pragma warning(push)
#pragma warning(disable: 4244)  // 'argument' : conversion from 'type1' to 'type2', possible loss of data
#pragma warning(disable: 4365)  // 'argument': conversion from '%1' to '%2', signed/unsigned mismatch
#pragma warning(disable: 28251) // Inconsistent annotation for '%1': this instance has no annotations.
#include "CUDA/helper_math.h"
#pragma warning(pop)

// CUDA
#include <cuda_fp16.h>

// C++
#define _USE_MATH_DEFINES
#include <math.h>

namespace Tracer
{
	//--------------------------------------------------------------------------------------------------------------------------
	// extra constants
	//--------------------------------------------------------------------------------------------------------------------------
	constexpr float Pi               = 3.1415926535897932f;
	constexpr float TwoPi            = 6.2831853071795865f; // 2 * Pi
	constexpr float FourPi           = 12.566370614359173f; // 4 * Pi
	constexpr float HalfPi           = 1.5707963267948966f; // Pi / 2
	constexpr float PiOverFour       = 0.7853981633974483f; // Pi / 4
	constexpr float RcpPi            = 0.3183098861837907f; // 1 / Pi
	constexpr float TwoOverPi        = 0.6366197723675813f; // 2 / Pi
	constexpr float FourOverPi       = 1.2732395447351627f; // 4 / Pi
	constexpr float RcpHalfPi        = 0.6366197723675813f; // 1 / (Pi/2)
	constexpr float RcpTwoPi         = 0.1591549430918953f; // 1 / (2 * Pi) = 0.5 / Pi
	constexpr float RcpFourPi        = 0.0795774715459477f; // 1 / (4 * Pi)
	constexpr float SqrtPi           = 1.7724538509055160f; // sqrt(Pi)
	constexpr float PiSquare         = 9.8696044010893586f; // Pi^2
	constexpr float FourPiSquare     = 39.478417604357434f; // 4 * Pi^2
	constexpr float RcpPiSquare      = 0.1013211836423378f; // 1 / (Pi^2) = (1 / Pi)^2
	constexpr float RcpFourPiSquare  = 0.0253302959105844f; // 1 / (4 * Pi^2)
	constexpr float FourOverPiSquare = 0.4052847345693511f; // 1 / (4 * Pi^2)
	constexpr float SqrtTwo          = 1.4142135623730950f; // sqrt(2)
	constexpr float RcpSqrtTwo       = 0.7071067811865475f; // 1 / sqrt(2) = sqrt(2) / 2
	constexpr float SqrtThree        = 1.7320508075688773f; // sqrt(3)
	constexpr float GoldenRatio      = 1.6180339887498948f; // (1 + sqrt(5)) / 2
	constexpr float Ln10             = 2.3025850929940457f; // ln(10)
	constexpr float Phi              = 1.6180339887498948f; // Golden Ratio

	constexpr float Epsilon          = 1e-3f;
	constexpr float DegToRad         = 3.14159265359f / 180.f;
	constexpr float RadToDeg         = 180.f / 3.14159265359f;



	//--------------------------------------------------------------------------------------------------------------------------
	// extension functions
	//--------------------------------------------------------------------------------------------------------------------------
	inline int2 operator -(const int2& a) { return make_int2(-a.x, -a.y); }
	inline int3 operator -(const int3& a) { return make_int3(-a.x, -a.y, -a.z); }
	inline int4 operator -(const int4& a) { return make_int4(-a.x, -a.y, -a.z, -a.w); }
	inline float2 operator -(const float2& a) { return make_float2(-a.x, -a.y); }
	inline float3 operator -(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
	inline float4 operator -(const float4& a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }



	inline bool operator == (const int2& a, const int2& b) { return (a.x == b.x) && (a.y == b.y); }
	inline bool operator == (const int3& a, const int3& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
	inline bool operator == (const int4& a, const int4& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); }
	inline bool operator == (const uint2& a, const uint2& b) { return (a.x == b.x) && (a.y == b.y); }
	inline bool operator == (const uint3& a, const uint3& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
	inline bool operator == (const uint4& a, const uint4& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); }
	inline bool operator == (const float2& a, const float2& b) { return (a.x == b.x) && (a.y == b.y); }
	inline bool operator == (const float3& a, const float3& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
	inline bool operator == (const float4& a, const float4& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); }



	inline bool operator != (const int2& a, const int2& b) { return !(a == b); }
	inline bool operator != (const int3& a, const int3& b) { return !(a == b); }
	inline bool operator != (const int4& a, const int4& b) { return !(a == b); }
	inline bool operator != (const uint2& a, const uint2& b) { return !(a == b); }
	inline bool operator != (const uint3& a, const uint3& b) { return !(a == b); }
	inline bool operator != (const uint4& a, const uint4& b) { return !(a == b); }
	inline bool operator != (const float2& a, const float2& b) { return !(a == b); }
	inline bool operator != (const float3& a, const float3& b) { return !(a == b); }
	inline bool operator != (const float4& a, const float4& b) { return !(a == b); }



	//--------------------------------------------------------------------------------------------------------------------------
	// vector math
	//--------------------------------------------------------------------------------------------------------------------------
	inline float3 RotateAroundAxis(float3 v, float3 axis, float radians)
	{
		// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
		const float cosAngle = cosf(radians);
		const float sinAngle = sinf(radians);
		return (v * cosAngle) + (cross(axis, v) * sinAngle) + (axis * dot(axis, v) * (1.f - cosAngle));
	}



	inline float3 SmallestAxis(const float3& v)
	{
		// squaring is faster than abs
		const float x2 = v.x * v.x;
		const float y2 = v.y * v.y;
		const float z2 = v.z * v.z;

		if(x2 < y2)
			return x2 < z2 ? make_float3(1, 0, 0) : make_float3(0, 0, 1);
		return y2 < z2 ? make_float3(0, 1, 0) : make_float3(0, 0, 1);
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// half4
	//--------------------------------------------------------------------------------------------------------------------------
	struct half4
	{
		half x;
		half y;
		half z;
		half w;
	};



	inline half4 make_half4(half x, half y, half z, half w)
	{
		half4 t;
		t.x = x;
		t.y = y;
		t.z = z;
		t.w = w;
		return t;
	}



	inline half4 make_half4(float x, float y, float z, float w)
	{
		half4 t;
		t.x = __float2half(x);
		t.y = __float2half(y);
		t.z = __float2half(z);
		t.w = __float2half(w);
		return t;
	}



	inline half4 make_half4(const float4& f4)
	{
		half4 t;
		t.x = __float2half(f4.x);
		t.y = __float2half(f4.y);
		t.z = __float2half(f4.z);
		t.w = __float2half(f4.w);
		return t;
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// float3x4
	//--------------------------------------------------------------------------------------------------------------------------
	struct alignas(16) float3x4
	{
		float3 x; float tx;
		float3 y; float ty;
		float3 z; float tz;
	};



	// creation
	inline float3x4 make_float3x4(float xx = 1, float xy = 0, float xz = 0,
								  float yx = 0, float yy = 1, float yz = 0,
								  float zx = 0, float zy = 0, float zz = 1,
								  float tx = 0, float ty = 0, float tz = 0)
	{
		float3x4 t;
		t.x = make_float3(xx, xy, xz);
		t.y = make_float3(yx, yy, yz);
		t.z = make_float3(zx, zy, zz);
		t.tx = tx;
		t.ty = ty;
		t.tz = tz;
		return t;
	}



	inline float3x4 make_float3x4(const float3& x, const float3& y, const float3& z, float3 w)
	{
		float3x4 t;
		t.x = x;
		t.y = y;
		t.z = z;
		t.tx = w.x;
		t.ty = w.y;
		t.tz = w.z;
		return t;
	}



	// transform
	inline float3x4 transform(const float3x4& a, const float3x4& b)
	{
		const float3 t = make_float3(a.tx, a.ty, a.tz);
		const float3 b_x = make_float3(b.x.x, b.y.x, b.z.x);
		const float3 b_y = make_float3(b.x.y, b.y.y, b.z.y);
		const float3 b_z = make_float3(b.x.z, b.y.z, b.z.z);

		return make_float3x4(
			dot(a.x, b_x),
			dot(a.x, b_y),
			dot(a.x, b_z),

			dot(a.y, b_x),
			dot(a.y, b_y),
			dot(a.y, b_z),

			dot(a.z, b_x),
			dot(a.z, b_y),
			dot(a.z, b_z),

			dot(t, b_x) + b.tx,
			dot(t, b_y) + b.ty,
			dot(t, b_z) + b.tz);
	}



	inline float4 transform(const float3x4& a, const float4& b)
	{
		return make_float4(
			dot(b, make_float4(a.x.x, a.y.x, a.z.x, a.tx)),
			dot(b, make_float4(a.x.y, a.y.y, a.z.y, a.ty)),
			dot(b, make_float4(a.x.z, a.y.z, a.z.z, a.tz)),
			b.w);
	}



	// rotate
	inline float3x4 rotate_x_3x4(float a)
	{
		const float s = sinf(a);
		const float c = cosf(a);

		float3x4 m = make_float3x4();
		m.y.y = c;
		m.y.z = s;
		m.z.y = -s;
		m.z.z = c;
		return m;
	}



	inline float3x4 rotate_y_3x4(float a)
	{
		const float s = sinf(a);
		const float c = cosf(a);

		float3x4 m = make_float3x4();
		m.x.x = c;
		m.x.z = -s;
		m.z.x = s;
		m.z.z = c;
		return m;
	}


	inline float3x4 rotate_z_3x4(float a)
	{
		const float s = sinf(a);
		const float c = cosf(a);

		float3x4 m = make_float3x4();
		m.x.x = c;
		m.x.y = s;
		m.y.x = -s;
		m.y.y = c;
		return m;
	}



	inline float3x4 rotate_3x4(float x, float y, float z)
	{
		const float cx = cosf(x);
		const float sx = sinf(x);
		const float cy = cosf(y);
		const float sy = sinf(y);
		const float cz = cosf(z);
		const float sz = sinf(z);

		return make_float3x4(
			cy * cz,
			sy*sx - cy*sz*cx,
			cy*sz*sx + sy*cx,
			sz,
			cz*cx,
			-cz*sx,
			-sy*cz,
			sy*sz*cx + cy*sx,
			-sy*sz*sx + cy*cx);
	}



	inline float3x4 rotate_3x4(const float3& euler)
	{
		return rotate_3x4(euler.x, euler.y, euler.z);
	}



	// scale
	inline float3x4 scale_3x4(float scale)
	{
		float3x4 m = make_float3x4();
		m.x.x = scale;
		m.y.y = scale;
		m.z.z = scale;
		return m;
	}



	inline float3x4 scale_3x4(float x, float y, float z)
	{
		float3x4 m = make_float3x4();
		m.x.x = x;
		m.y.y = y;
		m.z.z = z;
		return m;
	}



	inline float3x4 scale_3x4(const float3& scale)
	{
		return scale_3x4(scale.x, scale.y, scale.z);
	}



	// translate
	inline float3x4 translate_3x4(float x, float y, float z)
	{
		float3x4 m = make_float3x4();
		m.tx = x;
		m.ty = y;
		m.tz = z;
		return m;
	}



	inline float3x4 translate_3x4(const float3& t)
	{
		return translate_3x4(t.x, t.y, t.z);
	}



	// look at
	inline float3x4 look_at(const float3& from, const float3& to, const float3& up)
	{
		const float3 z = normalize(to - from);
		const float3 x = normalize(cross(up, z));
		const float3 y = normalize(cross(z, x));

		float3x4 m = make_float3x4();
		m.x = x;
		m.y = y;
		m.z = z;
		m.tx = from.x;
		m.ty = from.y;
		m.tz = from.z;
		return m;
	}



	// interpolation
	inline float3x4 lerp(const float3x4& a, const float3x4& b, float t)
	{
		const float3 ta = make_float3(a.tx, a.ty, a.tz);
		const float3 tb = make_float3(b.tx, b.ty, b.tz);
		return make_float3x4(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t), lerp(ta, tb, t));
	}



	// deteminant
	inline float determinant(const float3x4& a)
	{
		return (a.x.x * (a.y.y * a.z.z - a.y.z * a.z.y) +
				a.x.y * (a.y.z * a.z.x - a.y.x * a.z.z) +
				a.x.z * (a.y.x * a.z.y - a.y.y * a.z.x));
	}



	// inverse
	inline float3x4 inverse(const float3x4& a)
	{
		const float det3 = a.x.x * (a.y.y*a.z.z - a.y.z*a.z.y) - a.x.y * (a.y.x*a.z.z - a.y.z*a.z.x) + a.x.z * (a.y.x*a.z.y - a.y.y*a.z.x);
		const float inv_det3 = 1.0f / det3;

		const float inv00 = inv_det3 * (a.y.y*a.z.z - a.z.y*a.y.z);
		const float inv01 = inv_det3 * (a.x.z*a.z.y - a.z.z*a.x.y);
		const float inv02 = inv_det3 * (a.x.y*a.y.z - a.y.y*a.x.z);

		const float inv10 = inv_det3 * (a.y.z*a.z.x - a.z.z*a.y.x);
		const float inv11 = inv_det3 * (a.x.x*a.z.z - a.z.x*a.x.z);
		const float inv12 = inv_det3 * (a.x.z*a.y.x - a.y.z*a.x.x);

		const float inv20 = inv_det3 * (a.y.x*a.z.y - a.z.x*a.y.y);
		const float inv21 = inv_det3 * (a.x.y*a.z.x - a.z.y*a.x.x);
		const float inv22 = inv_det3 * (a.x.x*a.y.y - a.y.x*a.x.y);

		float3x4 result;
		result.x.x = inv00;
		result.x.y = inv01;
		result.x.z = inv02;

		result.y.x = inv10;
		result.y.y = inv11;
		result.y.z = inv12;

		result.z.x = inv20;
		result.z.y = inv21;
		result.z.z = inv22;

		result.tx = -inv00 * a.tx - inv01 * a.ty - inv02 * a.tz;
		result.ty = -inv10 * a.tx - inv11 * a.ty - inv12 * a.tz;
		result.tz = -inv20 * a.tx - inv21 * a.ty - inv22 * a.tz;
		return result;
	}



	// decompose
	inline void decompose(const float3x4& a, float3& pos, float3& euler, float3& scale)
	{
		pos = make_float3(a.tx, a.ty, a.tz);
		scale = make_float3(length(a.x), length(a.y), length(a.z));

		const float3 tx = normalize(a.x);
		const float3 ty = normalize(a.y);
		const float3 tz = normalize(a.z);
		if(ty.x > .998f)
		{
			euler = make_float3(0, atan2f(tx.z, tz.z), Pi * .5f);
		}
		else if(ty.x < -.998f)
		{
			euler = make_float3(0, atan2f(tx.z, tz.z), -Pi * .5f);
		}
		else
		{
			euler = make_float3(atan2f(-ty.z, ty.y), atan2f(-tz.x, tx.x), asinf(ty.x));
		}
	}



	// operators
	inline float4 operator * (const float3x4& a, const float4& b) { return transform(a, b); }
	inline float3x4 operator * (const float3x4& a, const float3x4& b) { return transform(a, b); }
	inline float3x4& operator *= (float3x4& a, const float3x4& b) { return a = transform(a, b); }
}
