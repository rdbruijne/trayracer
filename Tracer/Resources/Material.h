#pragma once

// Project
#include "Utility/LinearMath.h"

// C++
#include <memory>
#include <string>

namespace Tracer
{
	class Texture;
	class Material
	{
	public:
		explicit Material(const std::string& name);

		size_t TextureCount() const
		{
			return (mDiffuseMap ? 1 : 0);
		}

		std::string mName = "";

		// material properties
		float3 mDiffuse = make_float3(.5f, .5f, .5f);

		// textures
		std::shared_ptr<Texture> mDiffuseMap = nullptr;
	};
}
