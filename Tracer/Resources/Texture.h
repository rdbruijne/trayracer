#pragma once

// Project
#include "Utility/LinearMath.h"

// C++
#include <string>
#include <vector>

namespace Tracer
{
	class Texture
	{
	public:
		Texture() = default;
		explicit Texture(const std::string& name, const uint2& resolution, std::vector<uint32_t> pixels);

		std::string mName = "";
		uint2 mResolution = make_uint2(0, 0);
		std::vector<uint32_t> mPixels;
	};
}
