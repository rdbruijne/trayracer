#include "MainGui.h"

// Project
#include "FileIO/ModelFile.h"
#include "FileIO/SceneFile.h"
#include "FileIO/TextureFile.h"
#include "GUI/GuiHelpers.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Renderer/Sky.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Utility/Utility.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// ImGUI
#include "imgui/imgui.h"
#pragma warning(push)
#pragma warning(disable: 4201 4263 4264 4458 5027 5038)
#include "imGuIZMO.quat/imGuIZMOquat.h"
#pragma warning(pop)

// C++
#include <functional>

namespace Tracer
{
	class Texture;

	namespace
	{
		std::string ImportModelDialog()
		{
			static std::string filter = "";
			if(filter.empty())
			{
				const std::vector<FileInfo>& info = ModelFile::SupportedFormats();
				std::string extensions = "";
				for(auto f : info)
				{
					std::vector<std::string> extParts = Split(f.ext, ',');
					for(auto e : extParts)
						extensions += std::string(extensions.empty() ? "" : ";") + "*." + e;
				}

				const std::string& zeroString = std::string(1, '\0');
				filter = "Model files" + zeroString + extensions + zeroString;
			}

			std::string modelFile = "";
			OpenFileDialog(filter.c_str(), "Select a model file", true, modelFile);
			return modelFile;
		}



		std::string ImportTextureDialog()
		{
			static std::string filter = "";
			if(filter.empty())
			{
				const std::vector<FileInfo>& info = TextureFile::SupportedFormats();
				std::string extensions = "";
				for(auto f : info)
				{
					std::vector<std::string> extParts = Split(f.ext, ',');
					for(auto e : extParts)
						extensions += std::string(extensions.empty() ? "" : ";") + "*." + e;
				}

				const std::string& zeroString = std::string(1, '\0');
				filter = "Image files" + zeroString + extensions + zeroString;
			}

			std::string texFile = "";
			OpenFileDialog(filter.c_str(), "Select a texture file", true, texFile);
			return texFile;
		}



		inline float PerSec(uint64_t count, float elapsedMs)
		{
			return (count == 0 || elapsedMs == 0) ? 0 : (static_cast<float>(count) / (elapsedMs * 1e-3f));
		}
	}



	MainGui* const MainGui::Get()
	{
		static MainGui inst;
		return &inst;
	}



	void MainGui::DrawImpl()
	{
		if(!ImGui::Begin("Tray Racer", &mEnabled))
			return;

		// camera
		if(ImGui::CollapsingHeader("Camera"))
			CameraElements();

		// material
		if(ImGui::CollapsingHeader("Material"))
			MaterialElements();

		// renderer
		if(ImGui::CollapsingHeader("Renderer"))
			RendererElements();

		// scene
		if(ImGui::CollapsingHeader("Scene"))
			SceneElements();

		// statistics
		if(ImGui::CollapsingHeader("Sky"))
			SkyElements();

		// statistics
		if(ImGui::CollapsingHeader("Statistics"))
			StatisticsElements();

		// debug
		if(ImGui::CollapsingHeader("Debug"))
			DebugElements();

		ImGui::End();
	}



	void MainGui::CameraElements()
	{
		if(!GuiHelpers::camNode)
		{
			ImGui::Text("No camera node detected");
		}
		else
		{
			bool hasChanged = false;

			// transformation
			float pos[] = {GuiHelpers::camNode->Position().x, GuiHelpers::camNode->Position().y, GuiHelpers::camNode->Position().z};
			float target[] = {GuiHelpers::camNode->Target().x, GuiHelpers::camNode->Target().y, GuiHelpers::camNode->Target().z};
			float up[] = {GuiHelpers::camNode->Up().x, GuiHelpers::camNode->Up().y, GuiHelpers::camNode->Up().z};

			float aperture = GuiHelpers::camNode->Aperture();
			float distortion = GuiHelpers::camNode->Distortion();
			float focalDist = GuiHelpers::camNode->FocalDist();
			float fov = GuiHelpers::camNode->Fov() * RadToDeg;

			ImGui::BeginGroup();
			ImGui::Text("Transformation");
			hasChanged = ImGui::InputFloat3("Position", pos) || hasChanged;
			hasChanged = ImGui::InputFloat3("Target", target) || hasChanged;
			hasChanged = ImGui::InputFloat3("Up", up) || hasChanged;
			ImGui::Spacing();

			ImGui::Text("Lens");
			hasChanged = ImGui::SliderFloat("Aperture", &aperture, 0.f, 10.f) || hasChanged;
			hasChanged = ImGui::SliderFloat("Distortion", &distortion, 0.f, 10.f) || hasChanged;
			hasChanged = ImGui::SliderFloat("Focal dist", &focalDist, 1.f, 1e6f, "%.3f", 10.f) || hasChanged;
			hasChanged = ImGui::SliderFloat("Fov", &fov, 1.f, 179.f) || hasChanged;
			ImGui::EndGroup();

			if(hasChanged)
			{
				GuiHelpers::camNode->SetPosition(make_float3(pos[0], pos[1], pos[2]));
				GuiHelpers::camNode->SetTarget(make_float3(target[0], target[1], target[2]));
				GuiHelpers::camNode->SetUp(make_float3(up[0], up[1], up[2]));
				GuiHelpers::camNode->SetAperture(aperture);
				GuiHelpers::camNode->SetDistortion(distortion);
				GuiHelpers::camNode->SetFocalDist(focalDist);
				GuiHelpers::camNode->SetFov(fov * DegToRad);
			}
		}
	}



	void MainGui::DebugElements()
	{
		ImGui::Columns(2);

		// table header
		ImGui::Separator();
		ImGui::Text("Data");
		ImGui::NextColumn();
		ImGui::Text("Value");
		ImGui::NextColumn();
		ImGui::Separator();

		// data
		for(auto& kv : mDebugItems)
		{
			ImGui::Text(kv.first.c_str());
			ImGui::NextColumn();
			ImGui::Text(kv.second.c_str());
			ImGui::NextColumn();
		}
		ImGui::Columns();
	}



	void MainGui::MaterialElements()
	{
		if(mSelectedMaterial.expired() || mSelectedMaterial.use_count() == 0)
		{
			ImGui::Text("No material detected");
		}
		else
		{
			// texture resolution
			const ImVec2 textureDisplayRes = ImVec2(100, 100) * ImGui::GetIO().FontGlobalScale;

			// fetch material
			std::shared_ptr<Material> mat = mSelectedMaterial.lock();

			// name
			ImGui::Columns(2);
			ImGui::Text(mat->Name().c_str());
			ImGui::NextColumn();
			ImGui::Text("Name");
			ImGui::Columns(1);

			// properties
			for(size_t i = 0; i < magic_enum::enum_count<Material::PropertyIds>(); i++)
			{
				const Material::PropertyIds id = static_cast<Material::PropertyIds>(i);
				const std::string propName = ToString(id);

				ImGui::Spacing();
				ImGui::Spacing();
				ImGui::Columns(2);

				// color
				if(mat->IsColorEnabled(id))
				{
					const std::string colorName = "##" + propName;
					const ImGuiColorEditFlags colorFlags =
						ImGuiColorEditFlags_HDR |
						ImGuiColorEditFlags_Float |
						ImGuiColorEditFlags_PickerHueWheel;
					float3 c = mat->GetColor(id);
					if(ImGui::ColorEdit3(colorName.c_str(), reinterpret_cast<float*>(&c), colorFlags))
						mat->Set(id, c);
				}

				// texture
				if(mat->IsTextureEnabled(id))
				{
					const std::string texName = "##" + propName + "map";
					std::shared_ptr<Texture> tex = mat->GetTextureMap(id);

					if(!tex)
					{
						// load button
						const std::string buttonName = "Load texture" + texName;
						if(ImGui::Button(buttonName.c_str()))
						{
							std::string texFile = ImportTextureDialog();
							if(!texFile.empty())
								mat->Set(id, TextureFile::Import(GuiHelpers::scene, texFile));
						}
					}
					else
					{
						// display texture
						tex->MakeGlTex();
						const size_t texId = static_cast<size_t>(tex->GLTex()->ID());
						if(ImGui::ImageButton(reinterpret_cast<ImTextureID>(texId), textureDisplayRes))
						{
							std::string texFile = ImportTextureDialog();
							if(!texFile.empty())
								mat->Set(id, TextureFile::Import(GuiHelpers::scene, texFile));
						}

						// remove texture button
						const std::string buttonName = "X" + texName;
						ImGui::SameLine();
						if(ImGui::Button(buttonName.c_str()))
							mat->Set(id, nullptr);
					}
				}
				ImGui::NextColumn();

				// property name
				ImGui::Text(propName.c_str());

				ImGui::Columns(1);
			}
		}
	}



	void MainGui::RendererElements()
	{
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
				for(size_t i = 0; i < magic_enum::enum_count<RenderModes>(); i++)
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
			if(ImGui::SliderInt("Max depth", &maxDepth, 1, 16))
				GuiHelpers::renderer->SetMaxDepth(maxDepth);

			float aoDist = GuiHelpers::renderer->AODist();
			if(ImGui::SliderFloat("AO Dist", &aoDist, 0.f, 1e4f, "%.3f", 10.f))
				GuiHelpers::renderer->SetAODist(aoDist);

			float zDepthMax = GuiHelpers::renderer->ZDepthMax();
			if(ImGui::SliderFloat("Z-Depth max", &zDepthMax, 0.f, 1e4f, "%.3f", 10.f))
				GuiHelpers::renderer->SetZDepthMax(zDepthMax);

			// post
			if(GuiHelpers::window)
			{
				ImGui::Spacing();
				ImGui::Text("Post");

				Window::ShaderProperties shaderProps = GuiHelpers::window->PostShaderProperties();
				ImGui::SliderFloat("Exposure", &shaderProps.exposure, 0.f, 100.f, "%.3f", 10.f);
				ImGui::SliderFloat("Gamma", &shaderProps.gamma, 0.f, 4.f);
				GuiHelpers::window->SetPostShaderProperties(shaderProps);
			}

			// denoiser
			ImGui::Spacing();
			ImGui::Text("Denoiser");

			bool denoising = GuiHelpers::renderer->DenoisingEnabled();
			if(ImGui::Checkbox("Enabled", &denoising))
				GuiHelpers::renderer->SetDenoiserEnabled(denoising);

			int32_t denoiserSampleThreshold = GuiHelpers::renderer->DenoiserSampleThreshold();
			if(ImGui::SliderInt("Sample threshold", &denoiserSampleThreshold, 0, 100))
				GuiHelpers::renderer->SetDenoiserSampleThreshold(denoiserSampleThreshold);
		}
	}



	void MainGui::SceneElements()
	{
		Scene_Scene();
		ImGui::Spacing();
		ImGui::Spacing();

		Scene_Models();
		ImGui::Spacing();
		ImGui::Spacing();

		Scene_Instances();
	}



	void MainGui::SkyElements()
	{
		auto sky = GuiHelpers::scene->GetSky();

		bool enabled = sky->Enabled();
		if(ImGui::Checkbox("Enabled", &enabled))
			sky->SetEnabled(enabled);

		// sun
		ImGui::Text("Sun");

		bool drawSun = sky->DrawSun();
		if(ImGui::Checkbox("Draw sun", &drawSun))
			sky->SetDrawSun(drawSun);

		float3 sunDir = sky->SunDir();
		vec3 l(sunDir.x, sunDir.y, sunDir.z);
		if(ImGui::gizmo3D("##sunDir3D", l, 100, imguiGizmo::modeDirection))
			sky->SetSunDir(make_float3(l.x, l.y, l.z));
		if(ImGui::InputFloat3("Sun dir", reinterpret_cast<float*>(&l)))
			sky->SetSunDir(make_float3(l.x, l.y, l.z));

		float sunSize = sky->SunAngularDiameter();
		if(ImGui::SliderFloat("Sun size (arc minutes)", &sunSize, 1.f, 10800.f, "%.3f", 10.f))
			sky->SetSunAngularDiameter(sunSize);

		float sunIntensity = sky->SunIntensity();
		if(ImGui::SliderFloat("Sun intensity", &sunIntensity, 0.f, 1e6f, "%.3f", 10.f))
			sky->SetSunIntensity(sunIntensity);

		// ground
		ImGui::Text("Ground");

		float turbidity = sky->Turbidity();
		if(ImGui::SliderFloat("Turbidity", &turbidity, 1.f, 10.f))
			sky->SetTurbidity(turbidity);

	}



	void MainGui::StatisticsElements()
	{
#define SPACE						\
		for(int i = 0; i < 4; i++)	\
		{							\
			ImGui::Spacing();		\
			ImGui::NextColumn();	\
		}

#define ROW(s, ...)					\
		ImGui::Text(s);				\
		ImGui::NextColumn();		\
		ImGui::Text(__VA_ARGS__);	\
		ImGui::NextColumn();

#define GRAPH(arr, s, ...)								\
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

		if(!GuiHelpers::renderer)
		{
			ImGui::Text("No renderer detected");
		}
		else
		{
			// fetch stats
			const Renderer::RenderStats renderStats = GuiHelpers::renderer->Statistics();

			// init column layout
			ImGui::Columns(2);

			// table header
#if false
			ROW("Stat", "Value");
			ImGui::Separator();
			ImGui::Separator();
#endif

			// device
			const auto& devProps = GuiHelpers::renderer->CudaDeviceProperties();
			ROW("Device", devProps.name);
			SPACE;

			// kenel
			ROW("Kernel", ToString(GuiHelpers::renderer->RenderMode()).c_str());
			ROW("Samples","%d", GuiHelpers::renderer->SampleCount());

			SPACE;

			// times
			ROW("FPS", "%.1f", 1e3f / mFrameTimeMs);
			ROW("Frame time", "%.1f ms", mFrameTimeMs);
			SPACE;
			ROW("Primary rays", "%.1f ms", renderStats.primaryPathTimeMs);
			ROW("Secondary rays", "%.1f ms", renderStats.secondaryPathTimeMs);
			ROW("Deep rays", "%.1f ms", renderStats.deepPathTimeMs);
			ROW("Shadow rays", "%.1f ms", renderStats.shadowTimeMs);
			SPACE;
			ROW("Shade time", "%.1f ms", renderStats.shadeTimeMs);
			ROW("Denoise time", "%.1f ms", renderStats.denoiseTimeMs);
			SPACE;
			ROW("Build time", "%.1f ms", renderStats.buildTimeMs);
			ROW("Geometry build time", "%.1f ms", renderStats.geoBuildTimeMs);
			ROW("Material build time", "%.1f ms", renderStats.matBuildTimeMs);
			ROW("Sky build time", "%.1f ms", renderStats.skyBuildTimeMs);

			SPACE;

			// rays
			ROW("Rays", "%.1f M (%.1f M/s)", renderStats.pathCount * 1e-6, PerSec(renderStats.pathCount, mFrameTimeMs) * 1e-6);
			ROW("Primaries", "%.1f M (%.1f M/s)", renderStats.primaryPathCount * 1e-6, PerSec(renderStats.primaryPathCount, renderStats.primaryPathTimeMs) * 1e-6);
			ROW("Secondaries", "%.1f M (%.1f M/s)", renderStats.secondaryPathCount * 1e-6, PerSec(renderStats.secondaryPathCount, renderStats.secondaryPathTimeMs) * 1e-6);
			ROW("Deep", "%.1f M (%.1f M/s)", renderStats.deepPathCount * 1e-6, PerSec(renderStats.deepPathCount, renderStats.deepPathTimeMs) * 1e-6);
			ROW("Shadow", "%.1f M (%.1f M/s)", renderStats.shadowRayCount * 1e-6, PerSec(renderStats.shadowRayCount, renderStats.shadowTimeMs) * 1e-6);

			SPACE;

			// scene
			ROW("Instance count", "%lld", GuiHelpers::scene->InstanceCount());
			ROW("Model count", "%lld", GuiHelpers::scene->InstancedModelCount());
			ROW("Texture count", "%lld", GuiHelpers::scene->TextureCount());
			ROW("Triangle count", "%s", ThousandSeparators(GuiHelpers::scene->TriangleCount()).c_str());
			ROW("Unique triangle count", "%s", ThousandSeparators(GuiHelpers::scene->UniqueTriangleCount()).c_str());
			ROW("Lights", "%s", ThousandSeparators(GuiHelpers::scene->LightCount()).c_str());
			ROW("Unique lights", "%s", ThousandSeparators(GuiHelpers::scene->UniqueLightCount()).c_str());

			ImGui::Columns();
		}

#undef SPACE
#undef ROW
	}



	void MainGui::Scene_Scene()
	{
		ImGui::Columns(3, nullptr, false);

		// Load scene
		if(ImGui::Button("Load scene", ImVec2(ImGui::GetWindowWidth() * .3f, 0)))
		{
			std::string sceneFile;
			if(OpenFileDialog("Json\0*.json\0", "Select a scene file", true, sceneFile))
			{
				GuiHelpers::scene->Clear();
				SceneFile::Load(sceneFile, GuiHelpers::scene, GuiHelpers::camNode, GuiHelpers::renderer, GuiHelpers::window);
			}
		}
		ImGui::NextColumn();

		// Save scene
		if(ImGui::Button("Save scene", ImVec2(ImGui::GetWindowWidth() * .3f, 0)))
		{
			std::string sceneFile;
			if(SaveFileDialog("Json\0*.json\0", "Select a scene file", sceneFile))
				SceneFile::Save(sceneFile, GuiHelpers::scene, GuiHelpers::camNode, GuiHelpers::renderer, GuiHelpers::window);
		}
		ImGui::NextColumn();

		// Clear scene
		if(ImGui::Button("Clear scene", ImVec2(ImGui::GetWindowWidth() * .3f, 0)))
		{
			GuiHelpers::scene->Clear();
		}

		ImGui::Columns();
	}



	void MainGui::Scene_Models()
	{
		ImGui::Columns(2, nullptr, true);

		// Gather model names
		auto models = GuiHelpers::scene->Models();
		std::vector<const char*> modelNames;
		modelNames.reserve(models.size());
		for(auto m : models)
			modelNames.push_back(m->Name().c_str());

		// Model selection
		if(ImGui::ListBox("Models", &mSelectedModelIx, modelNames.data(), static_cast<int>(modelNames.size())))
			SelectModel(mSelectedModelIx);
		ImGui::NextColumn();

		// Import model
		if(ImGui::Button("Import"))
		{
			std::string modelFile = ImportModelDialog();
			if(!modelFile.empty())
			{
				std::shared_ptr<Model> model = ModelFile::Import(GuiHelpers::scene, modelFile);
				if(model)
					GuiHelpers::scene->Add(model);

				models = GuiHelpers::scene->Models();
				mSelectedModelIx = static_cast<int>(models.size() - 1);
				strcpy_s(mModelName, mNameBufferSize, models[mSelectedModelIx]->Name().c_str());
			}
		}

		if(mSelectedModelIx >= static_cast<int>(models.size()))
			mSelectedModelIx = 0;
		auto model = models.size() > 0 ? models[mSelectedModelIx] : nullptr;

		// delete
		if(ImGui::Button("Delete##delete_model"))
		{
			GuiHelpers::scene->Remove(model);
			models = GuiHelpers::scene->Models();
			SelectModel(0);
		}

		// create instance
		if(ImGui::Button("Create instance") && model)
		{
			GuiHelpers::scene->Add(std::make_shared<Instance>(model->Name(), model, make_float3x4()));
			strcpy_s(mInstanceName, mNameBufferSize, GuiHelpers::scene->Instances()[mSelectedInstanceIx]->Name().c_str());
		}

		// Properties
		if(ImGui::InputText("Name##model_name", mModelName, mNameBufferSize, ImGuiInputTextFlags_EnterReturnsTrue) && model && strlen(mModelName) > 0)
			model->SetName(mModelName);

		ImGui::Columns();
	}



	void MainGui::Scene_Instances()
	{
		ImGui::Columns(2, nullptr, true);

		// Gather model names
		auto instances = GuiHelpers::scene->Instances();
		std::vector<const char*> instanceNames;
		instanceNames.reserve(instances.size());
		for(auto m : instances)
			instanceNames.push_back(m->Name().c_str());

		// Instance selection
		if(ImGui::ListBox("Instances", &mSelectedInstanceIx, instanceNames.data(), static_cast<int>(instanceNames.size())))
			SelectInstance(mSelectedInstanceIx);
		ImGui::NextColumn();

		if(instances.size() > 0)
		{
			if(mSelectedInstanceIx >= static_cast<int>(instances.size()))
				mSelectedInstanceIx = 0;

			auto inst = instances[mSelectedInstanceIx];
			auto model = inst->GetModel();

			// Name
			if(ImGui::InputText("Name##inst_name", mInstanceName, mNameBufferSize, ImGuiInputTextFlags_EnterReturnsTrue) && inst && strlen(mInstanceName) > 0)
				inst->SetName(mInstanceName);

			ImGui::InputText("Model", model ? const_cast<char*>(model->Name().c_str()) : nullptr, model ? static_cast<int>(model->Name().length()) : 0, ImGuiInputTextFlags_ReadOnly);

			// transform
			float3 pos;
			float3 scale;
			float3 euler;
			decompose(inst->Transform(), pos, euler, scale);

			float p[] = { pos.x, pos.y, pos.z };
			float s[] = { scale.x, scale.y, scale.z };
			float e[] = { euler.x * RadToDeg, euler.y * RadToDeg, euler.z * RadToDeg };

			bool changed = false;
			if(ImGui::InputFloat3("Pos", p, 2, ImGuiInputTextFlags_EnterReturnsTrue))
			{
				pos = make_float3(p[0], p[1], p[2]);
				changed = true;
			}

			if(ImGui::InputFloat3("Scale", s, 2, ImGuiInputTextFlags_EnterReturnsTrue))
			{
				scale = make_float3(s[0], s[1], s[2]);
				changed = true;
			}

			if(ImGui::InputFloat3("Euler", e, 2, ImGuiInputTextFlags_EnterReturnsTrue))
			{
				euler = make_float3(e[0] * DegToRad, e[1] * DegToRad, e[2] * DegToRad);
				changed = true;
			}

			if(changed)
				inst->SetTransform(rotate_3x4(euler) * scale_3x4(scale) * translate_3x4(pos));


			// delete
			if(ImGui::Button("Delete##delete_instance"))
			{
				GuiHelpers::scene->Remove(inst);
				instances = GuiHelpers::scene->Instances();
				SelectInstance(0);
			}
		}

		ImGui::Columns();
	}



	void MainGui::SelectModel(int ix)
	{
		mSelectedModelIx = ix;
		const auto& models = GuiHelpers::scene->Models();
		if(static_cast<int>(models.size()) > mSelectedModelIx)
			strcpy_s(mModelName, mNameBufferSize, models[mSelectedModelIx]->Name().c_str());
		else
			mModelName[0] = '\0';
	}



	void MainGui::SelectInstance(int ix)
	{
		mSelectedInstanceIx = ix;
		const auto& instances = GuiHelpers::scene->Instances();
		if(static_cast<int>(instances.size()) > mSelectedInstanceIx)
			strcpy_s(mInstanceName, mNameBufferSize, instances[mSelectedInstanceIx]->Name().c_str());
		else
			mInstanceName[0] = '\0';
	}
}
