#pragma once

// Project
#include "GuiWindow.h"

// C++
#include <memory>

namespace Tracer
{
	class Material;
	class MaterialWindow : public GuiWindow
	{
	public:
		static MaterialWindow* const Get();

		std::weak_ptr<Material> selectedMaterial = {};

	private:
		void DrawImpl() final;
	};
}
