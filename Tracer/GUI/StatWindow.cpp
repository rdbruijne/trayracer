#include "StatWindow.h"

// Project
#include "Gui/GuiHelpers.h"
#include "Optix/Renderer.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
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

			ImGui::Separator();
			ImGui::Text("Stat");
			ImGui::NextColumn();
			ImGui::Text("Value");
			ImGui::NextColumn();
			ImGui::Separator();

			// frametime
			ImGui::Text("Frame time");
			ImGui::NextColumn();
			ImGui::Text("%.1f ms", mFrameTimeNs * 1e-3);
			ImGui::NextColumn();

			// FPS
			ImGui::Text("FPS");
			ImGui::NextColumn();
			ImGui::Text("%.1f", 1e6 / mFrameTimeNs);
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
