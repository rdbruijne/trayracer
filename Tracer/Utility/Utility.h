#pragma once

// c++
#include <string>

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
}
