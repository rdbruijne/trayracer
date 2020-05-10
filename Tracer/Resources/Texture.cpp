#include "Texture.h"

namespace Tracer
{
	Texture::Texture(const std::string& name, const uint2& resolution, std::vector<uint32_t> pixels) :
		Resource(name),
		mResolution(resolution),
		mPixels(pixels)
	{
	}
}
