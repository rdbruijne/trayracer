#pragma once

// c++
#include <string>

#define NO_COPY_ALLOWED(c)				\
	c(const c&) = delete;				\
	c& operator =(const c&) = delete;	\
	c(c&& a) = default;					\
	c& operator =(c&& a) = default;

namespace Tracer
{
	// string manipulation
	std::string format(const char* fmt, ...);
	std::string ToLower(const std::string& str);
	std::string ToUpper(const std::string& str);

	// file paths
	std::string Directory(const std::string& filePath);
	std::string FileName(const std::string& filePath);
	std::string FileExtension(const std::string& filePath);
	std::string FileNameWithoutExtension(const std::string& filePath);

	// file handling
	std::string ReadFile(const std::string filePath);
	void WriteFile(const std::string filePath, const std::string& text);
	bool FileExists(const std::string& filePath);

	// stringify
	std::string TimeString(int64_t elapsedNs);
	std::string ThousandSeparators(uint64_t val, const std::string& separator = ",");

	// Windows
	bool OpenFileDialog(const char* filter, const std::string& title, bool mustExist, std::string& result);
}
