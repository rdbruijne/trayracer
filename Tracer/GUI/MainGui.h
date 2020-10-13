#pragma once

// C++
#include <array>
#include <map>
#include <memory>
#include <string>

namespace Tracer
{
	class Material;
	class MainGui
	{
	public:
		static MainGui* const Get();

		// drawing
		inline void SetEnabled(bool enable) { mEnabled = enable; }
		inline bool IsEnabled() const { return mEnabled; }
		inline void Draw()
		{
			if(mEnabled)
				DrawImpl();
		}

		// debug
		inline void SetDebug(const std::string& name, const std::string& data) { mDebugItems[name] = data; }
		inline void UnsetDebug(const std::string& name) { mDebugItems.erase(name); }

		// material
		inline void SelectMaterial(std::weak_ptr<Material> material) { mSelectedMaterial = material; }

		// statistics
		inline void UpdateStats(float frameTimeMs, float buildTimeMs)
		{
			mFrameTimeMs = frameTimeMs;
			mBuildTimeMs = buildTimeMs;
		}

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

		// drawing
		bool mEnabled = false;

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

		float mFrameTimeMs = 0;
		float mBuildTimeMs = 0;
	};
}
