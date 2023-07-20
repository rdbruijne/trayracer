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
		explicit CameraNode(const float3& position, const float3& target, const float3& up, float fov);

		// position
		const float3& Position() const { return mPosition; }
		void SetPosition(const float3& position);

		// target
		const float3& Target() const { return mTarget; }
		void SetTarget(const float3& target);

		// up
		const float3& Up() const { return mUp; }
		void SetUp(const float3& up);

		// transform
		float3x4 Transform() const;
		void SetTransform(const float3x4& t);

		// aperture
		float Aperture() const { return mAperture; }
		void SetAperture(float aperture);

		// distortion
		float Distortion() const { return mDistortion; }
		void SetDistortion(float distortion);

		// focal dist
		float FocalDist() const { return mFocalDist; }
		void SetFocalDist(float dist);

		// fov
		float Fov() const { return mFov; }
		void SetFov(float fov);

		// bokeh
		int BokehSideCount() const { return mBokehSideCount; }
		void SetBokehSideCount(int count);

		float BokehRotation() const { return mBokehRotation; }
		void SetBokehRotation(float rotation);

		// flags
		uint32_t Flags() const { return mFlags; }
		void SetFlags(uint32_t flags) { mFlags = flags; }

	private:
		float3 mPosition = make_float3(0, 0, -1);
		float mAperture = 0;

		float3 mTarget = make_float3(0, 0, 0);
		float mDistortion = 0;

		float3 mUp = make_float3(0, 1, 0);
		float mFocalDist = 1e5f;

		float mFov = 1.57079633f;
		float mBokehRotation = 0;
		int mBokehSideCount = 0;
		uint32_t mFlags;
	};
}
