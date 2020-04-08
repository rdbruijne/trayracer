#include "RendererWindow.h"

// Project
#include "Optix/Renderer.h"
#include "Gui/GuiHelpers.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	void RendererWindow::Draw()
	{
		// get the camera
		Renderer* renderer = GuiHelpers::renderer;

		ImGui::Begin("Renderer", &mEnabled);
		if(!renderer)
		{
			ImGui::Text("No renderer node detected");
			ImGui::End();
			return;
		}

		//bool hasChanged = false;

		// render mode
		Renderer::RenderModes activeRenderMode = renderer->RenderMode();
		const std::string rmName = ToString(activeRenderMode);
		if(ImGui::BeginCombo("Render Mode", rmName.c_str()))
		{
			for(size_t i = 0; i <magic_enum::enum_count<Renderer::RenderModes>(); i++)
			{
				const Renderer::RenderModes mode = static_cast<Renderer::RenderModes>(i);
				const std::string itemName = ToString(mode);
				if(ImGui::Selectable(itemName.c_str(), mode == activeRenderMode))
					renderer->SetRenderMode(mode);
				if(mode == activeRenderMode)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		// kernel settings
		ImGui::BeginGroup();
		int maxDepth = renderer->MaxDepth();
		if(ImGui::SliderInt("Max depth", &maxDepth, 0, 16))
			renderer->SetMaxDepth(maxDepth);

		ImGui::BeginGroup();
		float aoDist = renderer->AODist();
		if(ImGui::SliderFloat("AO Dist", &aoDist, 0.f, 1e4f, "%.3f", 10.f))
			renderer->SetAODist(aoDist);

		float zDepthMax = renderer->ZDepthMax();
		if(ImGui::SliderFloat("Z-Depth max", &zDepthMax, 0.f, 1e4f, "%.3f", 10.f))
			renderer->SetZDepthMax(zDepthMax);
		ImGui::EndGroup();

		ImGui::End();
	}
}
