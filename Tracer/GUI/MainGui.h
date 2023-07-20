#pragma once

// Project
#include "GUI/GuiWindow.h"
#include "Utility/LinearMath.h"

// C++
#include <map>
#include <memory>
#include <string>

namespace Tracer
{
	class Instance;
	class Material;
	class Model;
	class MainGui : public GuiWindow
	{
	public:
		// debug
		inline void SetDebug(const std::string& name, const std::string& data) { mDebugItems[name] = data; }
		inline void UnsetDebug(const std::string& name) { mDebugItems.erase(name); }

		// material
		void SelectMaterial(std::weak_ptr<Material> material);

	private:
		void DrawImpl() override;

		// elements
		void CameraElements();
		void DebugElements();
		void MaterialElements();
		void PostElements();
		void RendererElements();
		void SceneElements();
		void SkyElements();
		void StatisticsElements();

		// scene
		void Scene_Scene();
		void Scene_Models();
		void Scene_Instances();

		void SelectModel(int ix);
		void SelectInstance(int ix);

		// debug
		std::map<std::string, std::string> mDebugItems;

		// renderer
		int2 mResolution = make_int2(-1, -1);

		// selection
		static constexpr int mNameBufferSize = 128;

		// material
		std::weak_ptr<Material> mSelectedMaterial = {};

		// model
		std::weak_ptr<Model> mSelectedModel = {};
		int mSelectedModelIx = std::numeric_limits<int>::max();
		char mModelName[mNameBufferSize] = {};

		// instance
		std::weak_ptr<Instance> mSelectedInstance = {};
		int mSelectedInstanceIx = std::numeric_limits<int>::max();
		char mInstanceName[mNameBufferSize] = {};
		char mInstanceModelName[mNameBufferSize] = {};
	};
}
