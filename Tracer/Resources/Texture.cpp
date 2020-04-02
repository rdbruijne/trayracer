#include "Texture.h"

namespace Tracer
{
	Texture::Texture(const std::string& name, const uint2& resolution, std::vector<uint32_t> pixels) :
		mName(name),
		mResolution(resolution),
		mPixels(pixels)
	{
	}
}
