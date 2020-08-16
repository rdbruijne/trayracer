#pragma once

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Model;
	class Scene;
	class Texture;
	class Importer
	{
	public:
		struct Format
		{
			std::string name = "";
			std::string description = "";
			std::string ext = "";
		};
		static const std::vector<Format>& SupportedModelFormats();
		static const std::vector<Format>& SupportedTextureFormats();

		static std::shared_ptr<Texture> ImportTexture(Scene* scene, const std::string& textureFile, const std::string& importDir = "");
		static std::shared_ptr<Model> ImportModel(Scene* scene, const std::string& filePath, const std::string& name = "");
	};
}
