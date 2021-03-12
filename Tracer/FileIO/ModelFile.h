#pragma once

// Project
#include "FileIO/FileInfo.h"

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Model;
	class Scene;
	class ModelFile
	{
	public:
		static const std::vector<FileInfo>& SupportedFormats();
		static bool Supports(const std::string filePath);

		static std::shared_ptr<Model> Import(Scene* scene, const std::string& filePath, const std::string& name = "");
	};
}
