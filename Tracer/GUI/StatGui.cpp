#include "StatGui.h"

// Project
#include "GUI/GuiHelpers.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"

// ImGUI
#include "imgui/imgui.h"

#define USE_GRAPHS	false

namespace Tracer
{
	namespace
	{
		inline float PerSec(uint64_t count, float elapsedMs)
		{
			return (count == 0 || elapsedMs == 0) ? 0 : (static_cast<float>(count) / (elapsedMs * 1e-3f));
		}
	}



	StatGui* const StatGui::Get()
	{
		static StatGui inst;
		return &inst;
	}



	void StatGui::DrawImpl()
	{
#if USE_GRAPHS
#define SPACE				\
	ImGui::Spacing();		\
	ImGui::NextColumn();	\
	ImGui::Spacing();		\
	ImGui::NextColumn();	\
	ImGui::Separator();		\
	ImGui::Spacing();		\
	ImGui::NextColumn();	\
	ImGui::Spacing();		\
	ImGui::NextColumn();
#else
#define SPACE						\
	for(int i = 0; i < 4; i++)		\
	{								\
		ImGui::Spacing();			\
		ImGui::NextColumn();		\
	}
#endif

#define ROW(s, ...)				\
	ImGui::Text(s);				\
	ImGui::NextColumn();		\
	ImGui::Text(__VA_ARGS__);	\
	ImGui::NextColumn();

#define GRAPH(arr, s, ...)							\
	ImGui::Text(s);									\
	ImGui::NextColumn();							\
	ImGui::Text(__VA_ARGS__);						\
	ImGui::NextColumn();							\
	ImGui::NextColumn();							\
	ImGui::PushItemWidth(-1);						\
	ImGui::PlotLines("", arr.data(),				\
		static_cast<int>(msGraphSize),				\
		static_cast<int>(mGraphIx),					\
		nullptr, 0,									\
		*std::max_element(arr.begin(), arr.end()));	\
	ImGui::PopItemWidth();							\
	ImGui::NextColumn();

		ImGui::Begin("Statistics", &mEnabled);

		if(!GuiHelpers::renderer)
		{
			ImGui::Text("No renderer detected");
		}
		else
		{
			// fetch stats
			const Renderer::RenderStats renderStats = GuiHelpers::renderer->Statistics();

#if USE_GRAPHS
			// update graph times
			mFramerates[mGraphIx]         = 1e3f / mFrameTimeMs;
			mFrameTimes[mGraphIx]         = mFrameTimeMs;
			mBuildTimes[mGraphIx]         = mBuildTimeMs;
			mPrimaryPathTimes[mGraphIx]   = renderStats.primaryPathTimeMs;
			mSecondaryPathTimes[mGraphIx] = renderStats.secondaryPathTimeMs;
			mDeepPathTimes[mGraphIx]      = renderStats.deepPathTimeMs;
			mShadowTimes[mGraphIx]        = renderStats.shadowTimeMs;
			mShadeTimes[mGraphIx]         = renderStats.shadeTimeMs;
			mDenoiseTimes[mGraphIx]       = renderStats.denoiseTimeMs;

			// update pathcounts
			mPathCounts[mGraphIx]          = PerSec(renderStats.pathCount, mFrameTimeMs) * 1e-6f;
			mPrimaryPathCounts[mGraphIx]   = PerSec(renderStats.primaryPathCount, mFrameTimeMs) * 1e-6f;
			mSecondaryPathCounts[mGraphIx] = PerSec(renderStats.secondaryPathCount, mFrameTimeMs) * 1e-6f;
			mDeepPathCounts[mGraphIx]      = PerSec(renderStats.deepPathCount, mFrameTimeMs) * 1e-6f;
			mShadowRayCounts[mGraphIx]     = PerSec(renderStats.shadowRayCount, mFrameTimeMs) * 1e-6f;

			// increment graph ix
			mGraphIx = (mGraphIx + 1) % msGraphSize;
#endif

			// init column layout
			ImGui::Columns(2);

			// table header
#if false
			ROW("Stat", "Value");
			ImGui::Separator();
			ImGui::Separator();
#endif

			// device
			auto devProps = GuiHelpers::renderer->CudaDeviceProperties();
			ROW("Device", devProps.name);
			SPACE;

			// kenel
			ROW("Kernel", ToString(GuiHelpers::renderer->RenderMode()).c_str());
			ROW("Samples","%d", GuiHelpers::renderer->SampleCount());

			SPACE;

			// times
#if USE_GRAPHS
			GRAPH(mFramerates, "FPS", "%.1f", 1e3f / mFrameTimeMs);
			GRAPH(mFrameTimes, "Frame time", "%.1f ms", mFrameTimeMs);
			GRAPH(mBuildTimes, "Scene build", "%.1f ms", mBuildTimeMs);
			GRAPH(mPrimaryPathTimes, "Primary rays", "%.1f ms", renderStats.primaryPathTimeMs);
			GRAPH(mSecondaryPathTimes, "Secondary rays", "%.1f ms", renderStats.secondaryPathTimeMs);
			GRAPH(mDeepPathTimes, "Deep rays", "%.1f ms", renderStats.deepPathTimeMs);
			GRAPH(mShadowTimes, "Shadow rays", "%.1f ms", renderStats.shadowTimeMs);
			GRAPH(mShadeTimes, "Shade time", "%.1f ms", renderStats.shadeTimeMs);
			GRAPH(mDenoiseTimes, "Denoise time", "%.1f ms", renderStats.denoiseTimeMs);
#else
			ROW("FPS", "%.1f", 1e3f / mFrameTimeMs);
			ROW("Frame time", "%.1f ms", mFrameTimeMs);
			ROW("Scene build", "%.1f ms", mBuildTimeMs);
			ROW("Primary rays", "%.1f ms", renderStats.primaryPathTimeMs);
			ROW("Secondary rays", "%.1f ms", renderStats.secondaryPathTimeMs);
			ROW("Deep rays", "%.1f ms", renderStats.deepPathTimeMs);
			ROW("Shadow rays", "%.1f ms", renderStats.shadowTimeMs);
			ROW("Shade time", "%.1f ms", renderStats.shadeTimeMs);
			ROW("Denoise time", "%.1f ms", renderStats.denoiseTimeMs);
#endif

			SPACE;

			// rays
#if USE_GRAPHS
			GRAPH(mPathCounts, "Rays", "%.1f M (%.1f M/s)", renderStats.pathCount * 1e-6, PerSec(renderStats.pathCount, mFrameTimeMs) * 1e-6);
			GRAPH(mPrimaryPathCounts, "Primaries", "%.1f M (%.1f M/s)", renderStats.primaryPathCount * 1e-6, PerSec(renderStats.primaryPathCount, renderStats.primaryPathTimeMs) * 1e-6);
			GRAPH(mSecondaryPathCounts, "Secondaries", "%.1f M (%.1f M/s)", renderStats.secondaryPathCount * 1e-6f, PerSec(renderStats.secondaryPathCount, renderStats.secondaryPathTimeMs) * 1e-6f);
			GRAPH(mDeepPathCounts, "Deep", "%.1f M (%.1f M/s)", renderStats.deepPathCount * 1e-6f, PerSec(renderStats.deepPathCount, renderStats.deepPathTimeMs) * 1e-6f);
			GRAPH(mShadowRayCounts, "Shadow", "%.1f M (%.1f M/s)", renderStats.shadowRayCount * 1e-6f, PerSec(renderStats.shadowRayCount, renderStats.shadowTimeMs) * 1e-6f);
#else
			ROW("Rays", "%.1f M (%.1f M/s)", renderStats.pathCount * 1e-6, PerSec(renderStats.pathCount, mFrameTimeMs) * 1e-6);
			ROW("Primaries", "%.1f M (%.1f M/s)", renderStats.primaryPathCount * 1e-6, PerSec(renderStats.primaryPathCount, renderStats.primaryPathTimeMs) * 1e-6);
			ROW("Secondaries", "%.1f M (%.1f M/s)", renderStats.secondaryPathCount * 1e-6, PerSec(renderStats.secondaryPathCount, renderStats.secondaryPathTimeMs) * 1e-6);
			ROW("Deep", "%.1f M (%.1f M/s)", renderStats.deepPathCount * 1e-6, PerSec(renderStats.deepPathCount, renderStats.deepPathTimeMs) * 1e-6);
			ROW("Shadow", "%.1f M (%.1f M/s)", renderStats.shadowRayCount * 1e-6, PerSec(renderStats.shadowRayCount, renderStats.shadowTimeMs) * 1e-6);
#endif

			SPACE;

			// scene
			ROW("Instance count", "%lld", GuiHelpers::scene->InstanceCount());
			ROW("Model count", "%lld", GuiHelpers::scene->InstancedModelCount());
			ROW("Triangle count", "%s", ThousandSeparators(GuiHelpers::scene->TriangleCount()).c_str());
			ROW("Unique triangle count", "%s", ThousandSeparators(GuiHelpers::scene->UniqueTriangleCount()).c_str());
			ROW("Lights", "%s", ThousandSeparators(GuiHelpers::scene->LightCount()).c_str());
			ROW("Unique lights", "%s", ThousandSeparators(GuiHelpers::scene->UniqueLightCount()).c_str());

			ImGui::Columns();

#undef SPACE
#undef ROW
		}

		ImGui::End();
	}
}
