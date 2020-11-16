#include "TextureFile.h"

// Project
#include "FileIO/BinaryFile.h"
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

#define TEXTURE_CHACHE_ID_0			'T'
#define TEXTURE_CHACHE_ID_1			'E'
#define TEXTURE_CHACHE_ID_2			'X'
#define TEXTURE_CACHE_ID			{ TEXTURE_CHACHE_ID_0, TEXTURE_CHACHE_ID_1, TEXTURE_CHACHE_ID_2 }
#define TEXTURE_CACHE_VERSION		1

namespace Tracer
{
	namespace
	{
		//
		// Texture cache
		//
		bool SaveToCache(std::shared_ptr<Texture> tex, const std::string& filePath)
		{
			// create file
			const std::string cacheFile = BinaryFile::GenFilename(filePath);
			BinaryFile f(cacheFile, BinaryFile::FileMode::Write);

			// write the header
			BinaryFile::Header header = { TEXTURE_CACHE_ID, TEXTURE_CACHE_VERSION };
			f.Write(header);

			// write content
			f.Write(tex->Resolution());
			f.WriteVec(tex->Pixels());

			return true;
		}



		std::shared_ptr<Texture> LoadFromCache(Scene* scene, const std::string& filePath, const std::string& importDir = "")
		{
			// cache file name
			const std::string cacheFile = BinaryFile::GenFilename(filePath);

			// compare write times
			if(!FileExists(cacheFile) || FileLastWriteTime(filePath) > FileLastWriteTime(cacheFile))
				return nullptr;

			// open cache file
			BinaryFile f(cacheFile, BinaryFile::FileMode::Read);

			// check the header
			BinaryFile::Header header = f.Read<BinaryFile::Header>();
			if((header.type[0] != TEXTURE_CHACHE_ID_0) || (header.type[1] != TEXTURE_CHACHE_ID_1) ||
			   (header.type[2] != TEXTURE_CHACHE_ID_2) || (header.version != TEXTURE_CACHE_VERSION))
			{
				Logger::Error("Invalid cache file for \"%s\" (%s): found %c%c%c %d, expected %c%c%c %d",
							  filePath.c_str(), cacheFile.c_str(),
							  header.type[0], header.type[1], header.type[2], header.version,
							  TEXTURE_CHACHE_ID_0, TEXTURE_CHACHE_ID_1, TEXTURE_CHACHE_ID_2, TEXTURE_CACHE_VERSION);
				return nullptr;
			}

			// read content
			int2 res = f.Read<int2>();
			std::vector<float4> pixels = f.ReadVec<float4>();

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
		std::shared_ptr<Texture> tex = LoadFromCache(scene, globalPath, importDir);
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
