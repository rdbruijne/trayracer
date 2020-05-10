#pragma once

// Project
#include "Resources/Resource.h"
#include "Utility/LinearMath.h"

namespace Tracer
{
	class CameraNode : public Resource
	{
	public:
		CameraNode() = default;
		explicit CameraNode(const float3& position, const float3& target, const float3& up, const float fov);

		// position
		inline float3 Position() const { return mPosition; }
		void SetPosition(const float3& position);

		// target
		float3 Target() const { return mTarget; }
		void SetTarget(const float3& target);

		// up
		float3 Up() const { return mUp; }
		void SetUp(const float3& up);

		// fov
		float Fov() const { return mFov; }
		void SetFov(const float& fov);

	private:
		float3 mPosition = make_float3(0, 0, -1);
		float3 mTarget = make_float3(0, 0, 0);
		float3 mUp = make_float3(0, 1, 0);
		float mFov = 1.57079633f;
	};
}
