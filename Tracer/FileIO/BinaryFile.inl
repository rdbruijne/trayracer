namespace Tracer
{
	// Read
	template<typename TYPE>
	TYPE Tracer::BinaryFile::Read()
	{
		assert(mMode == FileMode::Read);

		if constexpr (std::is_same_v<TYPE, std::string>)
		{
			// read string length
			const size_t len = Read<size_t>();
			if(len == 0)
				return "";

			// read string
			assert(mHead + len * sizeof(char) <= mCapacity);
			std::string s(len, '\0');
			memcpy(s.data(), mBuffer + mHead, len * sizeof(char));
			mHead += len * sizeof(char);

			return s;
		}
		else
		{
			// read value
			assert(mHead + sizeof(TYPE) <= mCapacity);

			TYPE val;
			memcpy(&val, mBuffer + mHead, sizeof(TYPE));
			mHead += sizeof(TYPE);
			return val;
		}
	}



	template<typename TYPE>
	std::vector<TYPE> Tracer::BinaryFile::ReadVec()
	{
		assert(mMode == FileMode::Read);

		// read element count
		const size_t elemCount = Read<size_t>();

		// read the data
		assert(mHead + elemCount * sizeof(TYPE) <= mCapacity);
		std::vector<TYPE> v(elemCount);
		memcpy(v.data(), mBuffer + mHead, elemCount * sizeof(TYPE));
		mHead += elemCount * sizeof(TYPE);

		return std::move(v);
	}



	// Write
	template<typename TYPE>
	void Tracer::BinaryFile::Write(const TYPE& data)
	{
		assert(mMode == FileMode::Write);

		if constexpr (std::is_same_v<TYPE, std::string>)
		{
			// write string length
			Write(data.length());

			// check size
			while(mHead + (data.length() * sizeof(char)) >= mCapacity)
				Grow();

			// copy data
			memcpy(mBuffer + mHead, data.data(), data.length() * sizeof(char));
			mHead += data.length() * sizeof(char);
		}
		else
		{
			// check size
			if(mHead + sizeof(data) >= mCapacity)
				Grow();

			// copy data
			memcpy(mBuffer + mHead, &data, sizeof(data));
			mHead += sizeof(data);
		}
	}



	template<typename TYPE>
	void Tracer::BinaryFile::WriteVec(const std::vector<TYPE>& v)
	{
		assert(mMode == FileMode::Write);

		// write length
		Write(v.size());

		// check size
		while(mHead + v.size() * sizeof(TYPE) >= mCapacity)
			Grow();

		// copy data
		memcpy(mBuffer + mHead, v.data(), v.size() * sizeof(TYPE));
		mHead += v.size() * sizeof(TYPE);
	}
}
