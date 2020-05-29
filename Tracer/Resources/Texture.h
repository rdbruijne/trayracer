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
		~Texture();

		std::string Path() { return mPath; }
		const std::string& Path() const { return mPath; }

		uint2 Resolution() const { return mResolution; }
		const std::vector<uint32_t> Pixels() const { return mPixels; }

		// build
		void Build();

		// build info
		cudaArray_t CudaArray() const { return mCudaArray; }
		cudaTextureObject_t CudaObject() const { return mCudaObject; }

	private:
		std::string mPath = "";
		uint2 mResolution = make_uint2(0, 0);
		std::vector<uint32_t> mPixels;

		// build data
		cudaArray_t mCudaArray = nullptr;
		cudaTextureObject_t mCudaObject = 0;
	};
}
