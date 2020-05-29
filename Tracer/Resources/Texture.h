#pragma once

// Project
#include "OpenGL/GLTexture.h"
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
		explicit Texture(const std::string& path, const int2& resolution, std::vector<float4> pixels);
		~Texture();

		std::string Path() { return mPath; }
		const std::string& Path() const { return mPath; }

		int2 Resolution() const { return mResolution; }
		const std::vector<float4> Pixels() const { return mPixels; }

		// build
		void Build();
		void MakeGlTex();

		// build info
		cudaArray_t CudaArray() const { return mCudaArray; }
		cudaTextureObject_t CudaObject() const { return mCudaObject; }

		// ImGui
		const GLTexture* GLTex() const { return mGlTexture; }

	private:
		std::string mPath = "";
		int2 mResolution = make_int2(0, 0);
		std::vector<float4> mPixels;

		// build data
		cudaArray_t mCudaArray = nullptr;
		cudaTextureObject_t mCudaObject = 0;

		// OpenGL (for ImGui)
		bool mRebuildGlTex = true;
		GLTexture* mGlTexture = nullptr;
	};
}
