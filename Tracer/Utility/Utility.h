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
	std::string GetFolder(const std::string& filePath);
	std::string GetFileName(const std::string& filePath);
	std::string GetFileExtension(const std::string& filePath);
	std::string GetFileNameWithoutExtension(const std::string& filePath);

	// file handling
	std::string ReadFile(const std::string filePath);
	void WriteFile(const std::string filePath, const std::string& text);
	bool FileExists(const std::string& filePath);
}
