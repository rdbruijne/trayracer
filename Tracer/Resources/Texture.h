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
		explicit Texture(const std::string& path, const uint2& resolution, std::vector<uint32_t> pixels);

		std::string Path() { return mPath; }
		const std::string& Path() const { return mPath; }

		uint2 Resolution() const { return mResolution; }
		const std::vector<uint32_t> Pixels() const { return mPixels; }

	private:
		std::string mPath = "";
		uint2 mResolution = make_uint2(0, 0);
		std::vector<uint32_t> mPixels;
	};
}
