namespace Tracer
{
	template<typename TYPE>
	void Tracer::CudaBuffer::Upload(const TYPE* data, size_t count, bool allowResize)
	{
		if(count == 0)
		{
			if(allowResize)
				Free();
		}
		else
		{
			const size_t size = count * sizeof(TYPE);
			if(allowResize && (!mPtr || mSize != size))
				Resize(size);

			assert(mPtr != nullptr);
			assert(mSize == size);
			CUDA_CHECK(cudaMemcpy(mPtr, static_cast<const void*>(data), size, cudaMemcpyHostToDevice));
		}
	}



	template<typename TYPE>
	void Tracer::CudaBuffer::Upload(const std::vector<TYPE>& data, bool allowResize)
	{
		Upload(data.data(), data.size(), allowResize);
	}




	template<typename TYPE>
	void Tracer::CudaBuffer::UploadAsync(const TYPE* data, size_t count, bool allowResize)
	{
		if(count == 0)
		{
			if(allowResize)
				Free();
		}
		else
		{
			const size_t size = count * sizeof(TYPE);
			if(allowResize && (!mPtr || mSize != size))
				Resize(size);

			assert(mPtr != nullptr);
			assert(mSize == size);
			CUDA_CHECK(cudaMemcpyAsync(mPtr, static_cast<const void*>(data), size, cudaMemcpyHostToDevice));
		}
	}



	template<typename TYPE>
	void Tracer::CudaBuffer::UploadAsync(const std::vector<TYPE>& data, bool allowResize)
	{
		UploadAsync(data.data(), data.size(), allowResize);
	}




	template<typename TYPE>
	void Tracer::CudaBuffer::Download(TYPE* data, size_t count) const
	{
		assert(mPtr != nullptr);
		assert(mSize == sizeof(TYPE) * count);
		CUDA_CHECK(cudaMemcpy(static_cast<void*>(data), mPtr, sizeof(TYPE) * count, cudaMemcpyDeviceToHost));
	}



	template<typename TYPE>
	void Tracer::CudaBuffer::Download(std::vector<TYPE>& data) const
	{
		Download(data.data(), data.size());
	}



	template<typename TYPE>
	void Tracer::CudaBuffer::DownloadAsync(TYPE* data, size_t count) const
	{
		assert(mPtr != nullptr);
		assert(mSize == sizeof(TYPE) * count);
		CUDA_CHECK(cudaMemcpyAsync(static_cast<void*>(data), mPtr, sizeof(TYPE) * count, cudaMemcpyDeviceToHost));
	}



	template<typename TYPE>
	void Tracer::CudaBuffer::DownloadAsync(std::vector<TYPE>& data) const
	{
		DownloadAsync(data.data(), data.size());
	}
}
