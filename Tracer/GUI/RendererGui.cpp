#include "RendererGui.h"

// Project
#include "FileIO/SceneFile.h"
#include "Gui/GuiHelpers.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"

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
		if(!mRenderer)
		{
			ImGui::Text("No renderer node detected");
		}
		else
		{
			// scene
			ImGui::Columns(3, nullptr, false);
			if(ImGui::Button("Load scene", ImVec2(ImGui::GetWindowWidth() * .3f, 0)))
			{
				std::string sceneFile;
				if(OpenFileDialog("Json\0*.json\0", "Select a scene file", true, sceneFile))
				{
					mScene->Clear();
					SceneFile::Load(sceneFile, mScene,mCamNode, mRenderer, mWindow);
				}
			}
			ImGui::NextColumn();

			if(ImGui::Button("Save scene", ImVec2(ImGui::GetWindowWidth() * .3f, 0)))
			{
				std::string sceneFile;
				if(OpenFileDialog("Json\0*.json\0", "Select a scene file", false, sceneFile))
					SceneFile::Save(sceneFile, mScene,mCamNode, mRenderer, mWindow);
			}
			ImGui::NextColumn();

			if(ImGui::Button("Clear scene", ImVec2(ImGui::GetWindowWidth() * .3f, 0)))
			{
				mScene->Clear();
			}

			ImGui::Columns();
			ImGui::Spacing();

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

			int multiSample = mRenderer->MultiSample();
			if(ImGui::SliderInt("Multi-sample", &multiSample, 1, Renderer::MaxTraceDepth))
				mRenderer->SetMultiSample(multiSample);

			int maxDepth = mRenderer->MaxDepth();
			if(ImGui::SliderInt("Max depth", &maxDepth, 0, 16))
				mRenderer->SetMaxDepth(maxDepth);

			float aoDist = mRenderer->AODist();
			if(ImGui::SliderFloat("AO Dist", &aoDist, 0.f, 1e4f, "%.3f", 10.f))
				mRenderer->SetAODist(aoDist);

			float zDepthMax = mRenderer->ZDepthMax();
			if(ImGui::SliderFloat("Z-Depth max", &zDepthMax, 0.f, 1e4f, "%.3f", 10.f))
				mRenderer->SetZDepthMax(zDepthMax);

			float3 skyColor = mRenderer->SkyColor();
			if(ImGui::ColorEdit3("Sky color", reinterpret_cast<float*>(&skyColor)))
				mRenderer->SetSkyColor(skyColor);

			// post
			if(mWindow)
			{
				ImGui::Spacing();
				ImGui::Text("Post");

				Window::ShaderProperties shaderProps = mWindow->PostShaderProperties();
				ImGui::SliderFloat("Exposure", &shaderProps.exposure, 0.f, 100.f, "%.3f", 10.f);
				ImGui::SliderFloat("Gamma", &shaderProps.gamma, 0.f, 4.f, "%.3f", 1.f);
				mWindow->SetPostShaderProperties(shaderProps);
			}

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
