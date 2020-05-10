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

		inline std::shared_ptr<Model> GetModel() const { return mModel; }

		inline const float3x4& Transform() const { return mTransform; }
		inline void SetTransform(const float3x4& transform) { mTransform = transform; MarkDirty(); }

	private:
		float3x4 mTransform = make_float3x4();
		std::shared_ptr<Model> mModel = nullptr;
	};
}
