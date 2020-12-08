#pragma once

// Project
#include "FileIO/FileInfo.h"

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Texture;
	class Scene;
	class TextureFile
	{
	public:
		static const std::vector<FileInfo>& SupportedFormats();
		static std::shared_ptr<Texture> Import(Scene* scene, const std::string& filePath, const std::string& importDir = "");
	};
}