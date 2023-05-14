#include "Texture.h"

// Project
#include "CUDA/CudaError.h"
#include "Utility/Filesystem.h"

// CUDA
#include <cuda_fp16.h>

namespace Tracer
{
	Texture::Texture(const std::string& path, const int2& resolution, const std::vector<half4>& pixels) :
		Resource(FileNameExt(path)),
		mPath(path),
		mResolution(resolution),
		mPixels(pixels)
	{
	}



	Texture::Texture(const std::string& path, const int2& resolution, const std::vector<float4>& pixels) :
		Resource(FileNameExt(path)),
		mPath(path),
		mResolution(resolution)
	{
		// convert pixels to half
		mPixels.reserve(pixels.size());
		for(const float4& p : pixels)
			mPixels.push_back(make_half4(p));
	}



	Texture::Texture(const std::string& path, const int2& resolution, const std::vector<uint32_t>& pixels) :
		Resource(FileNameExt(path)),
		mPath(path),
		mResolution(resolution)
	{
		// convert pixels to half
		mPixels.reserve(pixels.size());
		for(const uint32_t& p : pixels)
		{
			const float a = static_cast<float>((p >> 24) & 0xFF) / 255.f;
			const float r = static_cast<float>((p >> 16) & 0xFF) / 255.f;
			const float g = static_cast<float>((p >>  8) & 0xFF) / 255.f;
			const float b = static_cast<float>((p >>  0) & 0xFF) / 255.f;
			mPixels.push_back(make_half4(r, g, b, a));
		}
	}



	Texture::~Texture()
	{
		if(mCudaArray)
			CUDA_CHECK(cudaFreeArray(mCudaArray));
		if(mCudaObject)
			CUDA_CHECK(cudaDestroyTextureObject(mCudaObject));
		if(mGlTexture)
			delete mGlTexture;
	}



	void Texture::Build()
	{
		std::lock_guard<std::mutex> l(mMutex);

		// dirty check
		if(!IsDirty())
			return;

		// mark out of sync
		MarkOutOfSync();

		// mark clean
		MarkClean();
	}



	void Texture::Upload(Renderer* /*renderer*/)
	{
		std::lock_guard<std::mutex> l(mMutex);

		// sync check
		if(!IsOutOfSync())
			return;

		// cleanup
		if(mCudaArray)
			CUDA_CHECK(cudaFreeArray(mCudaArray));
		if(mCudaObject)
			CUDA_CHECK(cudaDestroyTextureObject(mCudaObject));

		// check for validity
		if(!IsValid())
		{
			mCudaArray = nullptr;
			mCudaObject = 0;
			mRebuildGlTex = true;
			return;
		}

		// dimensions
		constexpr uint32_t numComponents = 4;
		const uint32_t width  = static_cast<uint32_t>(mResolution.x);
		const uint32_t height = static_cast<uint32_t>(mResolution.y);
		const uint32_t pitch  = static_cast<uint32_t>(width) * numComponents * sizeof(half);

		// texture descriptor
		cudaTextureDesc texDesc     = {};
		texDesc.addressMode[0]      = cudaAddressModeWrap;
		texDesc.addressMode[1]      = cudaAddressModeWrap;
		texDesc.filterMode          = cudaFilterModeLinear;
		texDesc.readMode            = cudaReadModeElementType;
		texDesc.sRGB                = 0;
		texDesc.borderColor[0]      = 1.0f;
		texDesc.normalizedCoords    = 1;
		texDesc.maxAnisotropy       = 1;
		texDesc.mipmapFilterMode    = cudaFilterModeLinear;
		texDesc.minMipmapLevelClamp = 0;
		texDesc.maxMipmapLevelClamp = 99;

		// create channel descriptor
		const cudaChannelFormatDesc channelDesc = cudaCreateChannelDescHalf4();

		// upload pixels
		CUDA_CHECK(cudaMallocArray(&mCudaArray, &channelDesc, width, height));
		CUDA_CHECK(cudaMemcpy2DToArrayAsync(mCudaArray, 0, 0, mPixels.data(), pitch, pitch, height, cudaMemcpyHostToDevice));

		// resource descriptor
		cudaResourceDesc resourceDesc = {};
		resourceDesc.resType = cudaResourceTypeArray;
		resourceDesc.res.array.array = mCudaArray;

		// texture object
		CUDA_CHECK(cudaCreateTextureObject(&mCudaObject, &resourceDesc, &texDesc, nullptr));

		// signal for OpenGL texture rebuild
		mRebuildGlTex = true;

		// mark synced
		MarkSynced();
	}



	void Texture::CreateGLTex()
	{
		if(mRebuildGlTex)
		{
			// delete existing texture if resized
			if(mGlTexture && mGlTexture->Resolution() != mResolution)
			{
				delete mGlTexture;
				mGlTexture = nullptr;
			}

			// create new if no texture exists
			if(!mGlTexture)
				mGlTexture = new GLTexture(mResolution, GLTexture::Types::Half4);

			// upload pixels
			mGlTexture->Upload(mPixels);

			mRebuildGlTex = false;
		}
	}



	void Texture::DestroyGLTex()
	{
		delete mGlTexture;
		mGlTexture = nullptr;
		mRebuildGlTex = true;
	}
}
