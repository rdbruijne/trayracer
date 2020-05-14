#pragma once

// C++
#include <memory>
#include <string>

namespace Tracer
{
	class Model;
	class Importer
	{
	public:
		static std::shared_ptr<Model> ImportModel(const std::string& filePath);
	};
}
