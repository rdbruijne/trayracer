#pragma once

// Project
#include "OpenGL/GLTexture.h"
#include "Resources/Resource.h"
#include "Utility/LinearMath.h"

// C++
#include <string>
#include <mutex>
#include <vector>

namespace Tracer
{
	class Texture : public Resource
	{
	public:
		Texture() = default;
		explicit Texture(const std::string& path, const int2& resolution, const std::vector<half4>& pixels);
		explicit Texture(const std::string& path, const int2& resolution, const std::vector<float4>& pixels);
		~Texture();

		Texture& operator =(const Texture& t) = delete;

		std::string Path() { return mPath; }
		const std::string& Path() const { return mPath; }

		const int2& Resolution() const { return mResolution; }
		const std::vector<half4>& Pixels() const { return mPixels; }
		bool IsValid() const { return mResolution.x > 0 && mResolution.y > 0 && mPixels.size() > 0; }

		// build
		void Build();

		// build info
		cudaArray_t CudaArray() const { return mCudaArray; }
		cudaTextureObject_t CudaObject() const { return mCudaObject; }

		// GL Tex
		void CreateGLTex();
		void DestroyGLTex();
		const GLTexture* GLTex() const { return mGlTexture; }

	private:
		std::string mPath = "";
		int2 mResolution = make_int2(0, 0);
		std::vector<half4> mPixels;

		// build data
		std::mutex mBuildMutex;
		cudaArray_t mCudaArray = nullptr;
		cudaTextureObject_t mCudaObject = 0;

		// OpenGL (for ImGui)
		bool mRebuildGlTex = true;
		GLTexture* mGlTexture = nullptr;
	};
}
