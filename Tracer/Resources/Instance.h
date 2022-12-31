#pragma once

// Project
#include "Resources/Resource.h"
#include "Utility/LinearMath.h"

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Model;
	class Instance : public Resource
	{
	public:
		Instance() = default;
		explicit Instance(const std::string& name, std::shared_ptr<Model> model, const float3x4& transform = make_float3x4());

		// model
		inline std::shared_ptr<Model> GetModel() const { return mModel; }

		// transform
		inline const float3x4& Transform() const { return mTransform; }
		inline void SetTransform(const float3x4& transform)
		{
			mTransform = transform;
			decompose(mTransform, mPos, mEuler, mScale);
			MarkDirty();
		}

		// decomposed transform
		inline void DecomposedTransform(float3& pos, float3& euler, float3& scale) const
		{
			pos = mPos;
			euler = mEuler;
			scale = mScale;
		}
		inline void SetDecomposedTransform(const float3& pos, const float3& euler, const float3& scale)
		{
			mPos = pos;
			mEuler = euler;
			mScale = scale;
			mTransform = rotate_3x4(euler) * scale_3x4(scale) * translate_3x4(pos);
			MarkDirty();
		}

	private:
		float3 mPos = make_float3(0);
		float3 mEuler = make_float3(0);
		float3 mScale = make_float3(1);
		float3x4 mTransform = make_float3x4();
		std::shared_ptr<Model> mModel = nullptr;
	};
}
