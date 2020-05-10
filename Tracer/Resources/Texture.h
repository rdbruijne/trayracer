#pragma once

// Project
#include "Resources/Resource.h"
#include "Utility/LinearMath.h"

// C++
#include <string>
#include <vector>

namespace Tracer
{
	class Texture : public Resource
	{
	public:
		Texture() = default;
		explicit Texture(const std::string& name, const uint2& resolution, std::vector<uint32_t> pixels);

		uint2 Resolution() const { return mResolution; }
		const std::vector<uint32_t> Pixels() const { return mPixels; }

	private:
		uint2 mResolution = make_uint2(0, 0);
		std::vector<uint32_t> mPixels;
	};
}
