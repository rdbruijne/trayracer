#pragma once

// C++
#include <cassert>
#include <string>
#include <vector>

namespace Tracer
{
	class BinaryFile
	{
		// disable copying
		BinaryFile(const BinaryFile&) = delete;
		BinaryFile& operator =(const BinaryFile&) = delete;

	public:
		enum class FileMode
		{
			Read,
			Write
		};

		struct Header
		{
			char type[3] = {};
			uint8_t version = 0;
		};

		BinaryFile() = delete;
		explicit BinaryFile(const std::string& path, FileMode mode);
		~BinaryFile();

		// flush to disk
		void Flush();

		// Read
		template<typename TYPE> TYPE Read();
		template<typename TYPE> std::vector<TYPE> ReadVec();

		// Write
		template<typename TYPE> void Write(const TYPE& data);
		template<typename TYPE> void WriteVec(const std::vector<TYPE>& v);

		// helper functions
		static std::string CachedFilePath(const std::string& path);

	private:
		void Grow();

		const std::string mPath = "";
		const FileMode mMode;

		size_t mCapacity = 0;
		size_t mHead = 0;
		char* mBuffer = nullptr;
	};
}

#include "BinaryFile.inl"
