#include "StatWindow.h"

// Project
#include "Gui/GuiHelpers.h"
#include "Renderer/Renderer.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	namespace
	{
		inline float PerSec(uint64_t count, float elapsedMs)
		{
			return (count == 0 || elapsedMs == 0) ? 0 : (static_cast<float>(count) / (elapsedMs * 1e-3f));
		}
	}



	StatWindow* const StatWindow::Get()
	{
		static StatWindow inst;
		return &inst;
	}



	void StatWindow::DrawImpl()
	{
#define SPACE						\
	for(int i = 0; i < 4; i++)		\
	{								\
		ImGui::Spacing();			\
		ImGui::NextColumn();		\
	}

#define ROW(s, ...)				\
	ImGui::Text(s);				\
	ImGui::NextColumn();		\
	ImGui::Text(__VA_ARGS__);	\
	ImGui::NextColumn();

		ImGui::Begin("Statistics", &mEnabled);

		if(!mRenderer)
		{
			ImGui::Text("No renderer detected");
		}
		else
		{
			ImGui::Columns(2);

			// fetch stats
			const Renderer::RenderStats renderStats = mRenderer->Statistics();

			// table header
			ImGui::Separator();
			ROW("Stat", "Value");
			ImGui::Separator();

			// kenel
			ROW("Kernel", ToString(mRenderer->RenderMode()).c_str());
			ROW("Samples","%d", mRenderer->SampleCount());

			SPACE;

			// times
			ROW("FPS", "%.1f", 1e3f / mFrameTimeMs);
			ROW("Frame time", "%.1f ms", mFrameTimeMs);
			ROW("Scene build", "%.1f ms", mBuildTimeMs);
			ROW("Primary rays", "%.1f ms", renderStats.primaryPathTimeMs);
			ROW("Secondary rays", "%.1f ms", renderStats.secondaryPathTimeMs);
			ROW("Deep rays", "%.1f ms", renderStats.deepPathTimeMs);
			ROW("Shade time", "%.1f ms", renderStats.shadeTimeMs);
			ROW("Denoise time", "%.1f ms", renderStats.denoiseTimeMs);

			SPACE;

			// rays
			ROW("Rays", "%.1f M (%.1f M/s)", renderStats.pathCount * 1e-6, PerSec(renderStats.pathCount, mFrameTimeMs) * 1e-6);
			ROW("Primaries", "%.1f M (%.1f M/s)", renderStats.primaryPathCount * 1e-6, PerSec(renderStats.primaryPathCount, renderStats.primaryPathTimeMs) * 1e-6);
			ROW("Secondaries", "%.1f M (%.1f M/s)", renderStats.secondaryPathCount * 1e-6, PerSec(renderStats.secondaryPathCount, renderStats.secondaryPathTimeMs) * 1e-6);
			ROW("Deep", "%.1f M (%.1f M/s)", renderStats.deepPathCount * 1e-6, PerSec(renderStats.deepPathCount, renderStats.deepPathTimeMs) * 1e-6);

#undef SPACE
#undef ROW
		}

		ImGui::End();
	}
}
