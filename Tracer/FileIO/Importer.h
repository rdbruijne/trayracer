#pragma once

// C++
#include <memory>
#include <string>

namespace Tracer
{
	class Model;
	class Texture;
	class Importer
	{
	public:
		static std::shared_ptr<Texture> ImportTexture(const std::string& textureFile, const std::string& importDir = "");
		static std::shared_ptr<Model> ImportModel(const std::string& filePath, const std::string& name = "");
	};
}
