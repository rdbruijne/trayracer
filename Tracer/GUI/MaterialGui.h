#pragma once

// Project
#include "BaseGui.h"

// C++
#include <functional>
#include <memory>

namespace Tracer
{
	class Material;
	class Scene;
	class Texture;
	class MaterialGui : public BaseGui
	{
	public:
		static MaterialGui* const Get();

		std::weak_ptr<Material> mSelectedMaterial = {};
		Scene* mScene = nullptr;

	private:
		void DrawImpl() final;
		void DrawTexture(const std::string& name, std::function<std::shared_ptr<Texture>()> getTex, std::function<void(std::shared_ptr<Texture>)> setTex);
	};
}
