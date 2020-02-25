#pragma once

// Project
#include "Utility/LinearMath.h"

namespace Tracer
{
	struct CameraNode
	{
		CameraNode() = default;

		explicit CameraNode(const float3& position, const float3& target, const float3& up, const float fov) :
			Position(position),
			Target(target),
			Up(up),
			Fov(fov)
		{
		}

		float3 Position = make_float3(0, 0, -1);
		float3 Target = make_float3(0, 0, 0);
		float3 Up = make_float3(0, 1, 0);
		float Fov = 1.57079633f;
	};
}
