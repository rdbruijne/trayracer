#include "Utility.h"

// C++
#include <algorithm>
#include <assert.h>
#include <cctype>
#include <fstream>

// Windows
#include "Utility/WindowsLean.h"

namespace Tracer
{
	std::string format(const char* fmt, ...)
	{
		size_t n = strlen(fmt);
		int final_n = -1;

		std::string str;
		std::unique_ptr<char[]> formatted;
		va_list ap;
		do
		{
			n *= 2;
			formatted.reset(new char[n + 1]);
			va_start(ap, fmt);
			final_n = vsnprintf_s(formatted.get(), n - 1, n - 2, fmt, ap);
			va_end(ap);
		} while(final_n < 0);
		return std::string(formatted.get());
	}



	std::string ToLower(const std::string& str)
	{
		std::string s = str;
		std::transform(s.begin(), s.end(), s.begin(), [](char c)
		{
			return static_cast<char>(std::tolower(c));
		});
		return s;
	}



	std::string ToUpper(const std::string& str)
	{
		std::string s = str;
		std::transform(s.begin(), s.end(), s.begin(), [](char c)
		{
			return static_cast<char>(std::toupper(c));
		});
		return s;
	}



	std::string Directory(const std::string& filePath)
	{
		std::string folderPath = filePath;
		folderPath = folderPath.substr(0, folderPath.find_last_of('/'));
		folderPath = folderPath.substr(0, folderPath.find_last_of('\\'));
		return folderPath;
	}



	std::string FileName(const std::string& filePath)
	{
		std::string fileName = filePath;
		size_t i = fileName.find_last_of('/');
		if(i != std::string::npos)
			fileName = fileName.substr(i + 1);

		i = fileName.find_last_of('\\');
		if(i != std::string::npos)
			fileName = fileName.substr(i + 1);

		return fileName;
	}



	std::string FileExtension(const std::string& filePath)
	{
		const std::string fileName = FileName(filePath);
		const size_t dotIx = fileName.find_last_of('.');
		return dotIx == std::string::npos ? "" : fileName.substr(dotIx);
	}



	std::string FileNameWithoutExtension(const std::string& filePath)
	{
		const std::string fileName = FileName(filePath);
		return fileName.substr(0, fileName.find_last_of('.'));
	}



	std::string ReadFile(const std::string filePath)
	{
		std::ifstream fileStream(filePath);
		assert(fileStream.is_open());
		if(!fileStream.is_open())
			return "";

		fileStream.seekg(0, std::ios::end);
		const size_t fileSize = static_cast<size_t>(fileStream.tellg());
		fileStream.seekg(0, std::ios::beg);

		std::string fileContent;
		fileContent.reserve(fileSize);
		fileContent.assign(std::istreambuf_iterator<char>(fileStream), std::istreambuf_iterator<char>());

		return fileContent;
	}



	void WriteFile(const std::string filePath, const std::string& text)
	{
		std::ofstream fileStream;
		fileStream.open(filePath, std::ofstream::out | std::ofstream::trunc);
		assert(fileStream.is_open());
		fileStream.write(text.c_str(), static_cast<std::streamsize>(text.length()));
	}



	bool FileExists(const std::string& filePath)
	{
		DWORD fileAttrib = GetFileAttributesA(filePath.c_str());
		return (fileAttrib != INVALID_FILE_ATTRIBUTES && !(fileAttrib & FILE_ATTRIBUTE_DIRECTORY));
	}



	std::string TimeString(int64_t elapsedNs)
	{
		if(elapsedNs < 1'000)
			return format("%lld ns", elapsedNs);
		if(elapsedNs < 1'000'000)
			return format("%lld.%01lld us", elapsedNs / 1'000, (elapsedNs / 100) % 10);
		if(elapsedNs < 1'000'000'000)
			return format("%lld.%01lld ms", elapsedNs / 1'000'000, (elapsedNs / 100'000) % 10);
		if(elapsedNs < 60'000'000'000)
			return format("%lld.%01lld s", elapsedNs / 1'000'000'000, (elapsedNs / 100'000'000) % 10);

		// (hh:)mm:ss format
		const int64_t t2 = elapsedNs / 1'000'000'000;
		if(t2 < 3600)
			return format("%02lld:%02lld", t2 / 60, t2 % 60);
		return format("%lld:%02lld:%02lld", t2 / 3600, (t2 % 3600) / 60, t2 % 60);
	}



	std::string ThousandSeparators(uint64_t val, const std::string& separator)
	{
		std::string s = "";
		int i = 0;
		while(val > 0)
		{
			if(i && (i % 3) == 0)
				s = separator + s;
			s = "0123456789"[val % 10] + s;
			val /= 10;
			i++;
		}
		return s;
	}
}
