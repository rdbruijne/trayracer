#pragma once

// Project
#include "GUI/GuiWindow.h"

// C++
#include <map>
#include <memory>
#include <string>

namespace Tracer
{
	class Material;
	class MainGui : public GuiWindow
	{
	public:
		// debug
		inline void SetDebug(const std::string& name, const std::string& data) { mDebugItems[name] = data; }
		inline void UnsetDebug(const std::string& name) { mDebugItems.erase(name); }

		// material
		void SelectMaterial(std::weak_ptr<Material> material);

	private:
		void DrawImpl();

		// elements
		void CameraElements();
		void DebugElements();
		void MaterialElements();
		void RendererElements();
		void SceneElements();
		void SkyElements();
		void StatisticsElements();

		// debug
		std::map<std::string, std::string> mDebugItems;

		// material
		std::weak_ptr<Material> mSelectedMaterial = {};

		// scene
		void Scene_Scene();
		void Scene_Models();
		void Scene_Instances();

		void SelectModel(int ix);
		void SelectInstance(int ix);

		static constexpr int mNameBufferSize = 128;

		int mSelectedModelIx = 0;
		char mModelName[mNameBufferSize] = {};

		int mSelectedInstanceIx = 0;
		char mInstanceName[mNameBufferSize] = {};
	};
}
