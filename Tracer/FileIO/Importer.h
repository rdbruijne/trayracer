#pragma once

// C++
#include <memory>
#include <string>

namespace Tracer
{
	class Model;
	class Scene;
	class Texture;
	class Importer
	{
	public:
		static std::shared_ptr<Texture> ImportTexture(Scene* scene, const std::string& textureFile, const std::string& importDir = "");
		static std::shared_ptr<Model> ImportModel(Scene* scene, const std::string& filePath, const std::string& name = "");
	};
}
