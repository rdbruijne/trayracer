#include "Texture.h"

// Project
#include "CUDA/CudaError.h"
#include "Utility/Utility.h"

// CUDA
#include <cuda_fp16.h>

namespace Tracer
{
	Texture::Texture(const std::string& path, const int2& resolution, const std::vector<half4>& pixels) :
		Resource(FileName(path)),
		mPath(path),
		mResolution(resolution),
		mPixels(pixels)
	{
	}



	Texture::Texture(const std::string& path, const int2& resolution, const std::vector<float4>& pixels) :
		Resource(FileName(path)),
		mPath(path),
		mResolution(resolution)
	{
		// convert pixels to half
		mPixels.reserve(pixels.size());
		for(const float4& p : pixels)
			mPixels.push_back(make_half4(p));
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



	void Texture::MakeGlTex()
	{
		if(mRebuildGlTex)
		{
			// delete existing texture
			if(mGlTexture && mGlTexture->Resolution() != mResolution)
				delete mGlTexture;

			// create new if no texture exists
			if(!mGlTexture)
				mGlTexture = new GLTexture(mResolution, GLTexture::Types::Half4);

			// upload pixels
			mGlTexture->Upload(mPixels);

			mRebuildGlTex = false;
		}
	}



	void Texture::Build()
	{
		std::lock_guard<std::mutex> l(mBuildMutex);

		// dirty check
		if(!IsDirty())
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

		// create channel descriptor
		constexpr uint32_t numComponents = 4;
		const uint32_t width  = mResolution.x;
		const uint32_t height = mResolution.y;
		const uint32_t pitch  = width * numComponents * sizeof(half);
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDescHalf4();

		// upload pixels
		CUDA_CHECK(cudaMallocArray(&mCudaArray, &channelDesc, width, height));
		CUDA_CHECK(cudaMemcpy2DToArrayAsync(mCudaArray, 0, 0, mPixels.data(), pitch, pitch, height, cudaMemcpyHostToDevice));

		// resource descriptor
		cudaResourceDesc resourceDesc = {};
		resourceDesc.resType = cudaResourceTypeArray;
		resourceDesc.res.array.array = mCudaArray;

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

		// texture object
		CUDA_CHECK(cudaCreateTextureObject(&mCudaObject, &resourceDesc, &texDesc, nullptr));

		// signal for OpenGL texture rebuild
		mRebuildGlTex = true;

		// mark clean
		MarkClean();
	}
}
