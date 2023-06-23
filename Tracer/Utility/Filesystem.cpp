#include "Filesystem.h"

// C++
#include <assert.h>
#include <filesystem>
#include <fstream>
#include <stdlib.h>

// Windows
#include <shlwapi.h>

// undo defines
#ifdef CreateDirectory
#undef CreateDirectory
#endif

namespace Tracer
{
	void SplitPath(const std::string& path,
				   std::optional<std::reference_wrapper<std::string>> drive,
				   std::optional<std::reference_wrapper<std::string>> dir,
				   std::optional<std::reference_wrapper<std::string>> filename,
				   std::optional<std::reference_wrapper<std::string>> ext)
	{
		char drive_cstr[_MAX_DRIVE];
		char dir_cstr[_MAX_DIR];
		char fname_cstr[_MAX_FNAME];
		char ext_cstr[_MAX_EXT];
		const errno_t e = _splitpath_s(path.c_str(), drive_cstr, _MAX_DRIVE, dir_cstr, _MAX_DIR, fname_cstr, _MAX_FNAME, ext_cstr, _MAX_EXT);
		assert(e == 0);
		if(e != 0)
			return;

		if(drive)
			drive->get() = drive_cstr;
		if(dir)
			dir->get() = dir_cstr;
		if(filename)
			filename->get() = fname_cstr;
		if(ext)
			ext->get() = ext_cstr;
	}



	std::string MakePath(std::optional<std::reference_wrapper<const std::string>> drive,
						 std::optional<std::reference_wrapper<const std::string>> dir,
						 std::optional<std::reference_wrapper<const std::string>> filename,
						 std::optional<std::reference_wrapper<const std::string>> ext)
	{
		char newPath[MAX_PATH] = {};
		const char* driveCStr = drive ? drive->get().c_str() : nullptr;
		const char* dirCStr   = dir ? dir->get().c_str() : nullptr;
		const char* fileCStr  = filename ? filename->get().c_str() : nullptr;
		const char* extCStr   = ext ? ext->get().c_str() : nullptr;
		[[maybe_unused]]
		const errno_t e = _makepath_s(newPath, MAX_PATH, driveCStr, dirCStr, fileCStr, extCStr);
		assert(e == 0);
		return newPath;
	}



	std::string PathComponents(const std::string& path, bool drive, bool dir, bool filename, bool ext)
	{
		char drive_cstr[_MAX_DRIVE] = {};
		char dir_cstr[_MAX_DIR] = {};
		char fname_cstr[_MAX_FNAME] = {};
		char ext_cstr[_MAX_EXT] = {};
		const errno_t splitError = _splitpath_s(path.c_str(), drive_cstr, _MAX_DRIVE, dir_cstr, _MAX_DIR, fname_cstr, _MAX_FNAME, ext_cstr, _MAX_EXT);
		assert(splitError == 0);
		if(splitError != 0)
			return {};

		char newPath[MAX_PATH] = {};
		[[maybe_unused]]
		const errno_t makeError = _makepath_s(newPath, MAX_PATH, drive ? drive_cstr : nullptr, dir ? dir_cstr : nullptr,
											  filename ? fname_cstr : nullptr, ext ? ext_cstr : nullptr);
		assert(makeError == 0);
		return newPath;
	}



	std::string Drive(const std::string& path)
	{
		char drive[_MAX_DRIVE];
		[[maybe_unused]]
		const errno_t e = _splitpath_s(path.c_str(), drive, _MAX_DRIVE, nullptr, 0, nullptr, 0, nullptr, 0);
		assert(e == 0);
		return drive;
	}



	std::string Directory(const std::string& path)
	{
		char drive[_MAX_DRIVE];
		char dir[_MAX_DIR];
		[[maybe_unused]]
		const errno_t e = _splitpath_s(path.c_str(), drive, _MAX_DRIVE, dir, _MAX_DIR, nullptr, 0, nullptr, 0);
		assert(e == 0);

		char newPath[MAX_PATH] = {};
		[[maybe_unused]]
		const errno_t e2 = _makepath_s(newPath, MAX_PATH, drive, dir, nullptr, nullptr);
		assert(e2 == 0);
		return newPath;
	}



	std::string FileName(const std::string& path)
	{
		char fname[_MAX_FNAME];
		[[maybe_unused]]
		const errno_t e = _splitpath_s(path.c_str(), nullptr, 0, nullptr, 0, fname, _MAX_FNAME, nullptr, 0);
		assert(e == 0);
		return fname;
	}



	std::string FileExtension(const std::string& path)
	{
		char ext[_MAX_EXT];
		[[maybe_unused]]
		const errno_t e = _splitpath_s(path.c_str(), nullptr, 0, nullptr, 0, nullptr, 0, ext, _MAX_EXT);
		assert(e == 0);
		return ext;
	}



	std::string FileNameExt(const std::string& path)
	{
		char fname[_MAX_FNAME];
		char ext[_MAX_EXT];
		[[maybe_unused]]
		const errno_t e = _splitpath_s(path.c_str(), nullptr, 0, nullptr, 0, fname, _MAX_FNAME, ext, _MAX_EXT);
		assert(e == 0);

		char newPath[MAX_PATH] = {};
		[[maybe_unused]]
		const errno_t e2 = _makepath_s(newPath, MAX_PATH, nullptr, nullptr, fname, ext);
		assert(e2 == 0);
		return newPath;
	}



	std::string ReplaceExtension(const std::string& filePath, const std::string& newExtension)
	{
		std::string drive;
		std::string dir;
		std::string fname;
		std::string ext;
		SplitPath(filePath, drive, dir, fname, ext);
		return MakePath(drive, dir, fname, newExtension);
	}



	std::string CurrentDirectory()
	{
		char path[MAX_PATH] = {};
		GetCurrentDirectoryA(MAX_PATH, path);
		return std::string(path) + "\\";
	}



	std::string GlobalPath(const std::string& path)
	{
		char globalPath[MAX_PATH] = {};
		GetFullPathNameA(path.c_str(), MAX_PATH, globalPath, NULL);
		return strlen(globalPath) == 0 ? path : globalPath;
	}



	std::string RelativeFilePath(const std::string& path)
	{
		char relPath[MAX_PATH] = {};
		PathRelativePathToA(relPath, CurrentDirectory().c_str(), 0, path.c_str(), FILE_ATTRIBUTE_NORMAL);
		return strlen(relPath) == 0 ? path : relPath;
	}



	void CreateDirectory(const std::string& path)
	{
		if(!DirectoryExists(path))
			CreateDirectoryA(path.c_str(), nullptr);
	}



	bool DirectoryExists(const std::string& path)
	{
		DWORD fileAttrib = GetFileAttributesA(path.c_str());
		return (fileAttrib != INVALID_FILE_ATTRIBUTES && (fileAttrib & FILE_ATTRIBUTE_DIRECTORY));
	}



	std::string ReadFile(const std::string& filePath)
	{
		if(!FileExists(filePath))
			return "";

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



	std::vector<char> ReadBinaryFile(const std::string& filePath)
	{
		std::ifstream fileStream(filePath, std::ios::binary);
		assert(!fileStream.fail());
		if(fileStream.fail())
			return std::vector<char>();

		std::vector<char> data(std::istreambuf_iterator<char>(fileStream), {});

		assert(!fileStream.fail());
		if (fileStream.fail())
			return std::vector<char>();

		return data;
	}



	void WriteFile(const std::string& filePath, const std::string& text)
	{
		std::ofstream fileStream;
		fileStream.open(filePath, std::ofstream::out | std::ofstream::trunc);
		assert(fileStream.is_open());
		fileStream.write(text.c_str(), static_cast<std::streamsize>(text.length()));
	}



	void WriteFile(const std::string& filePath, const std::vector<std::string>& text)
	{
		std::ofstream fileStream;
		fileStream.open(filePath, std::ofstream::out | std::ofstream::trunc);
		assert(fileStream.is_open());
		for(const std::string& str : text)
			fileStream.write(str.c_str(), static_cast<std::streamsize>(str.length()));
	}



	bool FileExists(const std::string& filePath)
	{
		DWORD fileAttrib = GetFileAttributesA(filePath.c_str());
		return (fileAttrib != INVALID_FILE_ATTRIBUTES && !(fileAttrib & FILE_ATTRIBUTE_DIRECTORY));
	}



	uint64_t FileCreateTime(const std::string& filePath)
	{
		WIN32_FILE_ATTRIBUTE_DATA info = {};
		if(!GetFileAttributesExA(filePath.c_str(), GetFileExInfoStandard, &info))
			return 0;
		return static_cast<uint64_t>(info.ftCreationTime.dwHighDateTime) << 32 | static_cast<uint64_t>(info.ftCreationTime.dwLowDateTime);
	}



	uint64_t FileLastAccessTime(const std::string& filePath)
	{
		WIN32_FILE_ATTRIBUTE_DATA info = {};
		if(!GetFileAttributesExA(filePath.c_str(), GetFileExInfoStandard, &info))
			return 0;
		return static_cast<uint64_t>(info.ftLastAccessTime.dwHighDateTime) << 32 | static_cast<uint64_t>(info.ftLastAccessTime.dwLowDateTime);
	}



	uint64_t FileLastWriteTime(const std::string& filePath)
	{
		WIN32_FILE_ATTRIBUTE_DATA info = {};
		if(!GetFileAttributesExA(filePath.c_str(), GetFileExInfoStandard, &info))
			return 0;
		return static_cast<uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32 | static_cast<uint64_t>(info.ftLastWriteTime.dwLowDateTime);
	}



	uint64_t FileSize(const std::string& filePath)
	{
		WIN32_FILE_ATTRIBUTE_DATA info = {};
		if(!GetFileAttributesExA(filePath.c_str(), GetFileExInfoStandard, &info))
			return 0;
		return static_cast<uint64_t>(info.nFileSizeHigh) << 32 | static_cast<uint64_t>(info.nFileSizeLow);
	}
}
