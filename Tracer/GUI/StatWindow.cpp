#include "StatWindow.h"

// Project
#include "Gui/GuiHelpers.h"
#include "Renderer/Renderer.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	StatWindow* const StatWindow::Get()
	{
		static StatWindow inst;
		return &inst;
	}



	void StatWindow::DrawImpl()
	{
		ImGui::Begin("Statistics", &mEnabled);

		if(!mRenderer)
		{
			ImGui::Text("No renderer detected");
		}
		else
		{
			ImGui::Columns(2);

			// table header
			ImGui::Separator();
			ImGui::Text("Stat");
			ImGui::NextColumn();
			ImGui::Text("Value");
			ImGui::NextColumn();
			ImGui::Separator();

			// frametime
			ImGui::Text("Frame time");
			ImGui::NextColumn();
			ImGui::Text("%.1f ms", mFrameTimeNs * 1e-6);
			ImGui::NextColumn();

			// FPS
			ImGui::Text("FPS");
			ImGui::NextColumn();
			ImGui::Text("%.1f", 1e9 / mFrameTimeNs);
			ImGui::NextColumn();

			// Sample count
			ImGui::Text("Samples");
			ImGui::NextColumn();
			ImGui::Text("%d", mRenderer->SampleCount());
			ImGui::NextColumn();

			// Kernel
			ImGui::Text("Kernel");
			ImGui::NextColumn();
			ImGui::Text(ToString(mRenderer->RenderMode()).c_str());
			ImGui::NextColumn();
		}

		ImGui::End();
	}
}
