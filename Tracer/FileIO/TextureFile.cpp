#include "ModelFile.h"

// Project
#include "FileIO/TextureFile.h"
#include "Renderer/Scene.h"
#include "Resources/Material.h"
#include "Resources/Texture.h"
#include "Utility/Logger.h"
#include "Utility/Stopwatch.h"
#include "Utility/Utility.h"

// FreeImage
#include "FreeImage/FreeImage.h"

// C++
#include <filesystem>

namespace Tracer
{
	namespace
	{
		//
		// cache read helpers
		//
		template<typename TYPE>
		TYPE Read(FILE* f)
		{
			TYPE val;
			fread(&val, sizeof(TYPE), 1, f);
			return val;
		}



		template<>
		std::string Read<std::string>(FILE* f)
		{
			size_t len;
			fread(&len, sizeof(size_t), 1, f);
			std::string s(len, '\0');
			fread(s.data(), sizeof(char), len, f);
			return s;
		}



		template<typename TYPE>
		std::vector<TYPE> ReadVec(FILE* f)
		{
			size_t elemCount;
			fread(&elemCount, sizeof(size_t), 1, f);
			std::vector<TYPE> v(elemCount);
			fread(v.data(), sizeof(TYPE), elemCount, f);
			return v;
		}



		//
		// cache write helpers
		//
		template<typename TYPE>
		void Write(FILE* f, const TYPE& data)
		{
			fwrite(&data, sizeof(TYPE), 1, f);
		}



		template<>
		void Write(FILE* f, const std::string& s)
		{
			size_t len = s.length();
			fwrite(&len, sizeof(size_t), 1, f);
			fwrite(s.data(), sizeof(char), len, f);
		}



		template<typename TYPE>
		void WriteVec(FILE* f, const std::vector<TYPE>& v)
		{
			size_t elemCount = v.size();
			fwrite(&elemCount, sizeof(size_t), 1, f);
			fwrite(v.data(), sizeof(TYPE), elemCount, f);
		}



		//
		// Texture cache
		//
		bool SaveToCache(std::shared_ptr<Texture> tex, const std::string& textureFile)
		{
			// determine cache file name
			const size_t pathHash = std::hash<std::string>{}(textureFile);
			const std::string cacheFile = "cache/" + std::to_string(pathHash);

			// create directory & file
			std::filesystem::create_directory("cache");
			FILE* f = nullptr;
			if((fopen_s(&f, cacheFile.c_str(), "wb") != 0) || !f)
				return false;

			// write content
			Write(f, tex->Resolution());
			WriteVec(f, tex->Pixels());
			fclose(f);

			return true;
		}



		std::shared_ptr<Texture> LoadTextureFromCache(Scene* scene, const std::string& filePath, const std::string& importDir = "")
		{
			// determine cache file name
			const size_t pathHash = std::hash<std::string>{}(filePath);
			const std::string cacheFile = "cache/" + std::to_string(pathHash);

			// compare write times
			if(!FileExists(cacheFile) || FileLastWriteTime(filePath) > FileLastWriteTime(cacheFile))
				return nullptr;

			// open cache file
			FILE* f = nullptr;
			if((fopen_s(&f, cacheFile.c_str(), "rb") != 0) || !f)
				return nullptr;

			// read content
			int2 res = Read<int2>(f);
			std::vector<float4> pixels = ReadVec<float4>(f);
			return std::make_shared<Texture>(filePath, res, pixels);
		}
	}



	const std::vector<FileInfo>& TextureFile::SupportedFormats()
	{
		static std::vector<FileInfo> result;
		if(result.size() == 0)
		{
			const int formatCount = FreeImage_GetFIFCount();
			result.reserve(formatCount);
			for(int i = 0; i < formatCount; i++)
			{
				FREE_IMAGE_FORMAT fif = static_cast<FREE_IMAGE_FORMAT>(i);
				if(FreeImage_FIFSupportsReading(fif))
				{
					FileInfo fi;
					fi.name = FreeImage_GetFormatFromFIF(fif);
					fi.description = FreeImage_GetFIFDescription(fif);
					fi.ext = FreeImage_GetFIFExtensionList(fif);
					result.push_back(fi);
				}
			}
		}
		return result;
	}



	std::shared_ptr<Tracer::Texture> TextureFile::Import(Scene* scene, const std::string& filePath, const std::string& importDir)
	{
		std::string globalPath = filePath;
		if(!importDir.empty())
			globalPath = importDir + "/" + globalPath;
		globalPath = GlobalPath(globalPath);

		// check the scene for an existing texture
		if(std::shared_ptr<Texture> tex = scene->GetTexture(globalPath))
			return tex;

		// check the cache
		Stopwatch sw;
		std::shared_ptr<Texture> tex = LoadTextureFromCache(scene, globalPath, importDir);
		if(tex)
		{
			Logger::Info("Loaded \"%s\" from cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());
			return tex;
		}

		// load image
		FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(globalPath.c_str(), 0);
		if(fif == FIF_UNKNOWN)
			fif = FreeImage_GetFIFFromFilename(globalPath.c_str());

		FIBITMAP* tmp = FreeImage_Load(fif, globalPath.c_str());
		if(!tmp)
			return nullptr;

		//FIBITMAP* dib = FreeImage_ConvertTo32Bits(tmp);
		FIBITMAP* dib = FreeImage_ConvertToRGBAF(tmp);
		FreeImage_Unload(tmp);

		const uint32_t width  = FreeImage_GetWidth(dib);
		const uint32_t height = FreeImage_GetHeight(dib);
		std::vector<float4> pixels(width * height);
		for(uint32_t y = 0; y < height; y++)
		{
			const uint8_t* line = FreeImage_GetScanLine(dib, y);
			memcpy(pixels.data() + (y * width), line, width * sizeof(float4));
		}
		FreeImage_Unload(dib);

		// create texture
		tex = std::make_shared<Texture>(globalPath, make_int2(width, height), pixels);
		Logger::Info("Imported \"%s\" in %s", globalPath.c_str(), sw.ElapsedString().c_str());

		sw.Reset();
		SaveToCache(tex, globalPath);
		Logger::Debug("Saved \"%s\" to cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());

		return tex;
	}
}
