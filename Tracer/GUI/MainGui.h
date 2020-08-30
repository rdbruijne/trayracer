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
		char mModelName[mNameBufferSize];

		int mSelectedInstanceIx = 0;
		char mInstanceName[mNameBufferSize];

		// statistics
		static constexpr size_t msGraphSize = 128;

		float mFrameTimeMs = 0;
		float mBuildTimeMs = 0;

		size_t mGraphIx = 0;

		std::array<float, msGraphSize> mFramerates;
		std::array<float, msGraphSize> mFrameTimes;
		std::array<float, msGraphSize> mBuildTimes;
		std::array<float, msGraphSize> mPrimaryPathTimes;
		std::array<float, msGraphSize> mSecondaryPathTimes;
		std::array<float, msGraphSize> mDeepPathTimes;
		std::array<float, msGraphSize> mShadowTimes;
		std::array<float, msGraphSize> mShadeTimes;
		std::array<float, msGraphSize> mDenoiseTimes;

		std::array<float, msGraphSize> mPathCounts;
		std::array<float, msGraphSize> mPrimaryPathCounts;
		std::array<float, msGraphSize> mSecondaryPathCounts;
		std::array<float, msGraphSize> mDeepPathCounts;
		std::array<float, msGraphSize> mShadowRayCounts;
	};
}
