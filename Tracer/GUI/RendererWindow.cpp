#include "RendererWindow.h"

// Project
#include "Renderer/Renderer.h"
#include "Gui/GuiHelpers.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	RendererWindow* const RendererWindow::Get()
	{
		static RendererWindow inst;
		return &inst;
	}



	void RendererWindow::DrawImpl()
	{
		ImGui::Begin("Renderer", &mEnabled);
		if(!mRenderer)
		{
			ImGui::Text("No renderer node detected");
		}
		else
		{
			// render mode
			RenderModes activeRenderMode = mRenderer->RenderMode();
			const std::string rmName = ToString(activeRenderMode);
			if(ImGui::BeginCombo("Render Mode", rmName.c_str()))
			{
				for(size_t i = 0; i <magic_enum::enum_count<RenderModes>(); i++)
				{
					const RenderModes mode = static_cast<RenderModes>(i);
					const std::string itemName = ToString(mode);
					if(ImGui::Selectable(itemName.c_str(), mode == activeRenderMode))
						mRenderer->SetRenderMode(mode);
					if(mode == activeRenderMode)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}

			// kernel settings
			ImGui::Spacing();
			ImGui::Text("Settings");

			int maxDepth = mRenderer->MaxDepth();
			if(ImGui::SliderInt("Max depth", &maxDepth, 0, 16))
				mRenderer->SetMaxDepth(maxDepth);

			float aoDist = mRenderer->AODist();
			if(ImGui::SliderFloat("AO Dist", &aoDist, 0.f, 1e4f, "%.3f", 10.f))
				mRenderer->SetAODist(aoDist);

			float zDepthMax = mRenderer->ZDepthMax();
			if(ImGui::SliderFloat("Z-Depth max", &zDepthMax, 0.f, 1e4f, "%.3f", 10.f))
				mRenderer->SetZDepthMax(zDepthMax);

			// denoiser
			ImGui::Spacing();
			ImGui::Text("Denoiser");

			bool denoising = mRenderer->DenoisingEnabled();
			if(ImGui::Checkbox("Enabled", &denoising))
				mRenderer->SetDenoiserEnabled(denoising);

			int32_t denoiserSampleTreshold = mRenderer->DenoiserSampleTreshold();
			if(ImGui::SliderInt("Sample treshold", &denoiserSampleTreshold, 0, 100))
				mRenderer->SetDenoiserSampleTreshold(denoiserSampleTreshold);
		}

		ImGui::End();
	}
}
