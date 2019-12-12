#include "Utility.h"

// C++
#include <algorithm>
#include <assert.h>
#include <cctype>
#include <fstream>

// Windows
#include "WindowsLean.h"

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
		} while (final_n < 0);
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



	std::string GetFolder(const std::string& filePath)
	{
		std::string folderPath = filePath;
		folderPath = folderPath.substr(0, folderPath.find_last_of('/'));
		folderPath = folderPath.substr(0, folderPath.find_last_of('\\'));
		return folderPath;
	}



	std::string GetFileName(const std::string& filePath)
	{
		std::string fileName = filePath;
		size_t i = fileName.find_last_of('/');
		if (i != std::string::npos)
			fileName = fileName.substr(i + 1);

		i = fileName.find_last_of('\\');
		if (i != std::string::npos)
			fileName = fileName.substr(i + 1);

		return fileName;
	}



	std::string GetFileExtension(const std::string& filePath)
	{
		const std::string fileName = GetFileName(filePath);
		const size_t dotIx = fileName.find_last_of('.');
		return dotIx == std::string::npos ? "" : fileName.substr(dotIx);
	}



	std::string GetFileNameWithoutExtension(const std::string& filePath)
	{
		const std::string fileName = GetFileName(filePath);
		return fileName.substr(0, fileName.find_last_of('.'));
	}



	std::string ReadFile(const std::string filePath)
	{
		std::ifstream fileStream(filePath);
		assert(fileStream.is_open());
		if (!fileStream.is_open())
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
		DWORD fileAttrib = GetFileAttributes(filePath.c_str());
		return (fileAttrib != INVALID_FILE_ATTRIBUTES && !(fileAttrib & FILE_ATTRIBUTE_DIRECTORY));
	}
}
