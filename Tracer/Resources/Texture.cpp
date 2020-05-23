#include "Texture.h"

// Project
#include "Utility/Utility.h"

namespace Tracer
{
	Texture::Texture(const std::string& path, const uint2& resolution, std::vector<uint32_t> pixels) :
		Resource(FileName(path)),
		mPath(path),
		mResolution(resolution),
		mPixels(pixels)
	{
	}
}
