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



	inline bool operator == (const float2& a, const float2& b) { return (a.x == b.x) && (a.y == b.y); }
	inline bool operator == (const float3& a, const float3& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
	inline bool operator == (const float4& a, const float4& b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); }



	inline bool operator != (const float2& a, const float2& b) { return !(a == b); }
	inline bool operator != (const float3& a, const float3& b) { return !(a == b); }
	inline bool operator != (const float4& a, const float4& b) { return !(a == b); }



	//
	// vector math
	//
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



	//
	// float3x4
	//
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



	// scale
	inline float3x4 scale_3x4(float scale)
	{
		float3x4 m = make_float3x4();
		m.x.x = scale;
		m.y.y = scale;
		m.z.z = scale;
		return m;
	}



	inline float3x4 scale_3x4(const float3& scale)
	{
		float3x4 m = make_float3x4();
		m.x.x = scale.x;
		m.y.y = scale.y;
		m.z.z = scale.z;
		return m;
	}



	// translate
	inline float3x4 translate_3x4(const float3& t)
	{
		float3x4 m = make_float3x4();
		m.tx = t.x;
		m.ty = t.y;
		m.tz = t.z;
		return m;
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



	// operators
	inline float4 operator * (const float3x4& a, const float4& b) { return transform(a, b); }
	inline float3x4 operator * (const float3x4& a, const float3x4& b) { return transform(a, b); }
	inline float3x4& operator *= (float3x4& a, const float3x4& b) { return a = transform(a, b); }
}
