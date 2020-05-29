#include "Texture.h"

// Project
#include "Renderer/CudaError.h"
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



	Texture::~Texture()
	{
		if(mCudaArray)
			CUDA_CHECK(cudaFreeArray(mCudaArray));
		if(mCudaObject)
			CUDA_CHECK(cudaDestroyTextureObject(mCudaObject));
	}



	void Texture::Build()
	{
		if(!IsDirty())
			return;

		// cleanup
		if(mCudaArray)
			CUDA_CHECK(cudaFreeArray(mCudaArray));
		if(mCudaObject)
			CUDA_CHECK(cudaDestroyTextureObject(mCudaObject));

		// create channel descriptor
		constexpr uint32_t numComponents = 4;
		const uint32_t width  = mResolution.x;
		const uint32_t height = mResolution.y;
		const uint32_t pitch  = width * numComponents * sizeof(uint8_t);
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

		// upload pixels
		CUDA_CHECK(cudaMallocArray(&mCudaArray, &channelDesc, width, height));
		CUDA_CHECK(cudaMemcpy2DToArray(mCudaArray, 0, 0, mPixels.data(), pitch, pitch, height, cudaMemcpyHostToDevice));


		// resource descriptor
		cudaResourceDesc resourceDesc = {};
		resourceDesc.resType = cudaResourceTypeArray;
		resourceDesc.res.array.array = mCudaArray;

		// texture descriptor
		cudaTextureDesc texDesc     = {};
		texDesc.addressMode[0]      = cudaAddressModeWrap;
		texDesc.addressMode[1]      = cudaAddressModeWrap;
		texDesc.filterMode          = cudaFilterModeLinear;
		texDesc.readMode            = cudaReadModeNormalizedFloat;
		texDesc.sRGB                = 0;
		texDesc.borderColor[0]      = 1.0f;
		texDesc.normalizedCoords    = 1;
		texDesc.maxAnisotropy       = 1;
		texDesc.mipmapFilterMode    = cudaFilterModePoint;
		texDesc.minMipmapLevelClamp = 0;
		texDesc.maxMipmapLevelClamp = 99;

		// texture object
		CUDA_CHECK(cudaCreateTextureObject(&mCudaObject, &resourceDesc, &texDesc, nullptr));
	}

}
