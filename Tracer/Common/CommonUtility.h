#pragma once

// C++
#include <stdint.h>

// CUDA
#include "CUDA/helper_math.h"
#include "CUDA/random.h"

static __device__
inline void GenerateCameraRay(const float3& camPos, const float3& camForward, const float3& camSide, const float3& camUp,
							  float camFov, const float2& res, float3& O, float3& D, int2 pixelIndex, uint32_t& seed)
{
	const float2 index = make_float2(pixelIndex);
	const float2 jitter = make_float2(rnd(seed), rnd(seed));

	const float aspect = res.x / res.y;
	float2 screen = (((index + jitter) / res) * 2.0f) - make_float2(1, 1);
	screen.y /= aspect;

	const float tanFov2 = tanf(camFov / 2.0f);
	const float2 lensCoord = tanFov2 * screen;

	O = camPos;
	D = normalize(camForward + (lensCoord.x * camSide) + (lensCoord.y * camUp));
}
