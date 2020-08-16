#include "RendererGui.h"

// Project
#include "GUI/GuiHelpers.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	RendererGui* const RendererGui::Get()
	{
		static RendererGui inst;
		return &inst;
	}



	void RendererGui::DrawImpl()
	{
		ImGui::Begin("Renderer", &mEnabled);
		if(!GuiHelpers::renderer)
		{
			ImGui::Text("No renderer node detected");
		}
		else
		{
			// render mode
			RenderModes activeRenderMode = GuiHelpers::renderer->RenderMode();
			const std::string rmName = ToString(activeRenderMode);
			if(ImGui::BeginCombo("Render Mode", rmName.c_str()))
			{
				for(size_t i = 0; i <magic_enum::enum_count<RenderModes>(); i++)
				{
					const RenderModes mode = static_cast<RenderModes>(i);
					const std::string itemName = ToString(mode);
					if(ImGui::Selectable(itemName.c_str(), mode == activeRenderMode))
						GuiHelpers::renderer->SetRenderMode(mode);
					if(mode == activeRenderMode)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}

			// kernel settings
			ImGui::Spacing();
			ImGui::Text("Settings");

			int multiSample = GuiHelpers::renderer->MultiSample();
			if(ImGui::SliderInt("Multi-sample", &multiSample, 1, Renderer::MaxTraceDepth))
				GuiHelpers::renderer->SetMultiSample(multiSample);

			int maxDepth = GuiHelpers::renderer->MaxDepth();
			if(ImGui::SliderInt("Max depth", &maxDepth, 0, 16))
				GuiHelpers::renderer->SetMaxDepth(maxDepth);

			float aoDist = GuiHelpers::renderer->AODist();
			if(ImGui::SliderFloat("AO Dist", &aoDist, 0.f, 1e4f, "%.3f", 10.f))
				GuiHelpers::renderer->SetAODist(aoDist);

			float zDepthMax = GuiHelpers::renderer->ZDepthMax();
			if(ImGui::SliderFloat("Z-Depth max", &zDepthMax, 0.f, 1e4f, "%.3f", 10.f))
				GuiHelpers::renderer->SetZDepthMax(zDepthMax);

			float3 skyColor = GuiHelpers::renderer->SkyColor();
			if(ImGui::ColorEdit3("Sky color", reinterpret_cast<float*>(&skyColor)))
				GuiHelpers::renderer->SetSkyColor(skyColor);

			// post
			if(GuiHelpers::window)
			{
				ImGui::Spacing();
				ImGui::Text("Post");

				Window::ShaderProperties shaderProps = GuiHelpers::window->PostShaderProperties();
				ImGui::SliderFloat("Exposure", &shaderProps.exposure, 0.f, 100.f, "%.3f", 10.f);
				ImGui::SliderFloat("Gamma", &shaderProps.gamma, 0.f, 4.f, "%.3f", 1.f);
				GuiHelpers::window->SetPostShaderProperties(shaderProps);
			}

			// denoiser
			ImGui::Spacing();
			ImGui::Text("Denoiser");

			bool denoising = GuiHelpers::renderer->DenoisingEnabled();
			if(ImGui::Checkbox("Enabled", &denoising))
				GuiHelpers::renderer->SetDenoiserEnabled(denoising);

			int32_t denoiserSampleTreshold = GuiHelpers::renderer->DenoiserSampleTreshold();
			if(ImGui::SliderInt("Sample treshold", &denoiserSampleTreshold, 0, 100))
				GuiHelpers::renderer->SetDenoiserSampleTreshold(denoiserSampleTreshold);
		}

		ImGui::End();
	}
}
