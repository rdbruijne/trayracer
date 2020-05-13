#pragma once

// Project
#include "BaseGui.h"

// C++
#include <memory>

namespace Tracer
{
	class Material;
	class MaterialGui : public BaseGui
	{
	public:
		static MaterialGui* const Get();

		std::weak_ptr<Material> selectedMaterial = {};

	private:
		void DrawImpl() final;
	};
}
