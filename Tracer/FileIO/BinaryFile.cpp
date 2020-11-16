#include "BinaryFile.h"

// C++
#include <filesystem>
#include <fstream>

namespace Tracer
{

	BinaryFile::BinaryFile(const std::string& path, FileMode mode) :
		mPath(path),
		mMode(mode)
	{
		if(mode == FileMode::Read)
		{
			std::ifstream fileStream;
			fileStream.open(mPath, std::ifstream::in | std::ifstream::binary);
			assert(fileStream.is_open());

			fileStream.seekg(0, std::ios::end);
			mCapacity = static_cast<size_t>(fileStream.tellg());
			fileStream.seekg(0, std::ios::beg);

			mBuffer = static_cast<char*>(malloc(mCapacity));
			fileStream.read(mBuffer, mCapacity);
			fileStream.close();
		}
		else
		{
			mCapacity = 1ull << 20; // 1MB by default;
			mBuffer = static_cast<char*>(malloc(mCapacity));
		}
	}



	BinaryFile::~BinaryFile()
	{
		Flush();

		if(mBuffer)
			free(mBuffer);
	}



	void BinaryFile::Flush()
	{
		if(mMode == FileMode::Write)
		{
			// create the directory
			std::filesystem::create_directory(Directory(mPath));

			// dump the content
			std::ofstream fileStream;
			fileStream.open(mPath, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
			assert(fileStream.is_open());
			fileStream.write(mBuffer, static_cast<std::streamsize>(mHead));
		}
	}



	std::string BinaryFile::GenFilename(const std::string& path)
	{
		const size_t pathHash = std::hash<std::string>{}(path);
		return "cache/" + std::to_string(pathHash) + ".bin";
	}



	void BinaryFile::Grow()
	{
		mCapacity <<= 1;
		mBuffer = static_cast<char*>(realloc(mBuffer, mCapacity));
	}
}
