#pragma once

// c++
#include <string>

namespace Tracer
{
	/*!
	 * @brief Create a string from a format.
	 *
	 * `format("%s %i", "text", 42) -> "text 42"`
	 * @param[in] fmt C string that contains the text. It can optionally contain embedded _format specifiers_ that are replaced
	 *					by the values specified in subsequent additional arguments and formatted as requested.
	 * @param[in] ... Depending on the _format string_, the function may expect a sequence of additional arguments.
	 * @return The formatted string.
	 */
	std::string format(const char* fmt, ...);

	/*!
	 * @brief Convert a string to lowercase.
	 *
	 * `ToLower("Lorem Ipsum") -> "lorem ipsum"`
	 * @param[in] str The string to convert.
	 * @return The input string converted to lowercase.
	 */
	std::string ToLower(const std::string& str);

	/*!
	 * @brief Convert a string to uppercase.
	 *
	 * `ToUpper("Lorem Ipsum") -> "LOREM IPSUM"`
	 * @param[in] str The string to convert.
	 * @return The input string converted to uppercase.
	 */
	std::string ToUpper(const std::string& str);

	/*!
	 * @brief Get the folder path for a path.
	 *
	 * `GetFolder("C:/tmp/file.txt") -> "C:/tmp"`
	 * @param[in] filePath The source path.
	 * @return The parent folder path for the input path.
	 */
	std::string GetFolder(const std::string& filePath);

	/*!
	 * @brief Get the file name for a path.
	 *
	 * `GetFileName("C:/tmp/file.txt") -> "file.txt"`
	 * @param[in] filePath The source path.
	 * @return The file name for the input path.
	 */
	std::string GetFileName(const std::string& filePath);

	/*!
	 * @brief Get the file extension for a path.
	 *
	 * `GetFileName("C:/tmp/file.txt") -> ".txt"`
	 * @param[in] filePath The source path.
	 * @return The file extension for the input path, or an empty string if the input path does not contain an extension.
	 */
	std::string GetFileExtension(const std::string& filePath);

	/*!
	 * @brief Get the folder path for a path.
	 *
	 * `GetFileName("C:/tmp/file.txt") -> "file"`
	 * @param[in] filePath The source path.
	 * @return The parent folder path for the input path.
	 */
	std::string GetFileNameWithoutExtension(const std::string& filePath);

	/*!
	 * @brief Read the contents of a file.
	 * @param[in] filePath The path to the file to be read.
	 * @return The content of the file, or an empty string if the file does not exist.
	 */
	std::string ReadFile(const std::string filePath);

	/*!
	 * @brief Write text to a file.
	 *
	 * A new file will be created if none exists. An existing file will be overwritten.
	 * @param[in] filePath The path to the file to be written.
	 * @param[in] text The content of the file to be written.
	 */
	void WriteFile(const std::string filePath, const std::string& text);

	/*!
	 * @brief Check if a file exists.
	 * @param[in] filePath The path of the file to check.
	 * @return True if the file exists, false otherwise.
	 */
	bool FileExists(const std::string& filePath);
}
