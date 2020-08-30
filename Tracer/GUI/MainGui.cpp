#include "MainGui.h"

// Project
#include "FileIO/Importer.h"
#include "FileIO/SceneFile.h"
#include "GUI/GuiHelpers.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Utility/Utility.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// ImGUI
#include "imgui/imgui.h"

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
				const std::vector<Importer::Format>& formats = Importer::SupportedModelFormats();
				std::string extensions = "";
				for(auto f : formats)
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
				const std::vector<Importer::Format>& texFormats = Importer::SupportedTextureFormats();
				std::string extensions = "";
				for(auto f : texFormats)
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



		void ShowTexture(const std::string& name, std::function<std::shared_ptr<Texture>()> getTex, std::function<void(std::shared_ptr<Texture>)> setTex)
		{
			// #TODO: decent layout

			ImGui::Spacing();
			ImGui::Spacing();

			auto tex = getTex();
			if(!tex)
			{
				ImGui::Text("%s: N/A", name.c_str());
				const std::string buttonName = format("Load texture##%s", name.c_str());
				if(ImGui::Button(buttonName.c_str()))
				{
					std::string texFile = ImportTextureDialog();
					if(!texFile.empty())
						setTex(Importer::ImportTexture(GuiHelpers::scene, texFile));
				}
			}
			else
			{
				tex->MakeGlTex();

				// tex ID
				size_t texId = static_cast<size_t>(tex->GLTex()->ID());

				// resolution
				const ImVec2 res = ImVec2(100, 100);

				// display
				const float dpiScale = ImGui::GetIO().FontGlobalScale;
				ImGui::Text("%s", name.c_str(), tex->Name().c_str());
				ImGui::Columns(3);

				ImGui::Text(tex->Name().c_str());
				ImGui::NextColumn();

				if(ImGui::ImageButton(reinterpret_cast<ImTextureID>(texId), ImVec2(res.x * dpiScale, res.y * dpiScale)))
				{
					std::string texFile = ImportTextureDialog();
					if(!texFile.empty())
						setTex(Importer::ImportTexture(GuiHelpers::scene, texFile));
				}
				ImGui::NextColumn();

				// remove texture button
				const std::string buttonName = format("Remove##%s", name.c_str());
				if(ImGui::Button(buttonName.c_str()))
					setTex(nullptr);

				ImGui::Columns(1);
			}
		}
	}



	MainGui* const MainGui::Get()
	{
		static MainGui inst;
		return &inst;
	}



	void MainGui::DrawImpl()
	{
		// flags to enable scene tab by default
		static int flags = ImGuiTabItemFlags_SetSelected;
		bool sceneTab = true;

		if(!ImGui::Begin("Tray Racer", &mEnabled))
			return;

		if(!ImGui::BeginTabBar("Elements", ImGuiTabItemFlags_SetSelected))
			return;

		// camera
		if(ImGui::BeginTabItem("Camera"))
		{
			CameraElements();
			ImGui::EndTabItem();
		}

		// material
		if(ImGui::BeginTabItem("Material"))
		{
			MaterialElements();
			ImGui::EndTabItem();
		}

		// renderer
		if(ImGui::BeginTabItem("Renderer"))
		{
			RendererElements();
			ImGui::EndTabItem();
		}

		// scene
		if((flags == 0 && ImGui::BeginTabItem("Scene")) ||
			(flags != 0 && ImGui::BeginTabItem("Scene", &sceneTab, flags)))
		{
			SceneElements();
			ImGui::EndTabItem();
		}

		// statistics
		if(ImGui::BeginTabItem("Statistics"))
		{
			StatisticsElements();
			ImGui::EndTabItem();
		}

		// debug
		if(ImGui::BeginTabItem("Debug"))
		{
			DebugElements();
			ImGui::EndTabItem();
		}

		ImGui::EndTabBar();
		ImGui::End();

		flags = 0;
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
			float camPos[] = {GuiHelpers::camNode->Position().x, GuiHelpers::camNode->Position().y, GuiHelpers::camNode->Position().z};
			float camTarget[] = {GuiHelpers::camNode->Target().x, GuiHelpers::camNode->Target().y, GuiHelpers::camNode->Target().z};
			float camUp[] = {GuiHelpers::camNode->Up().x, GuiHelpers::camNode->Up().y, GuiHelpers::camNode->Up().z};
			float camFov = GuiHelpers::camNode->Fov() * RadToDeg;

			ImGui::BeginGroup();
			hasChanged = ImGui::InputFloat3("Position", camPos) || hasChanged;
			hasChanged = ImGui::InputFloat3("Target", camTarget) || hasChanged;
			hasChanged = ImGui::InputFloat3("Up", camUp) || hasChanged;
			hasChanged = ImGui::SliderFloat("Fov", &camFov, 1, 179) || hasChanged;
			ImGui::EndGroup();

			if(hasChanged)
			{
				GuiHelpers::camNode->SetPosition(make_float3(camPos[0], camPos[1], camPos[2]));
				GuiHelpers::camNode->SetTarget(make_float3(camTarget[0], camTarget[1], camTarget[2]));
				GuiHelpers::camNode->SetUp(make_float3(camUp[0], camUp[1], camUp[2]));
				GuiHelpers::camNode->SetFov(camFov * DegToRad);
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
	}



	void MainGui::MaterialElements()
	{
		if(mSelectedMaterial.expired() || mSelectedMaterial.use_count() == 0)
		{
			ImGui::Text("No material detected");
		}
		else
		{
			std::shared_ptr<Material> mat = mSelectedMaterial.lock();
			ImGui::Text(mat->Name().c_str());

			float3 diff = mat->Diffuse();
			if(ImGui::ColorEdit3("Diffuse", reinterpret_cast<float*>(&diff)))
				mat->SetDiffuse(diff);

			float3 em = mat->Emissive();
			if(ImGui::ColorEdit3("Emissive", reinterpret_cast<float*>(&em), ImGuiColorEditFlags_HDR))
				mat->SetEmissive(em);

			ShowTexture("Diffuse map", [=]() { return mat->DiffuseMap(); }, [=](std::shared_ptr<Texture> a) { mat->SetDiffuseMap(a); });
			ShowTexture("Normal map", [=]() { return mat->NormalMap(); }, [=](std::shared_ptr<Texture> a) { mat->SetNormalMap(a); });
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



	void MainGui::StatisticsElements()
	{
//#define USE_GRAPHS

#ifdef USE_GRAPHS
#define SPACE					\
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
		for(int i = 0; i < 4; i++)	\
		{							\
			ImGui::Spacing();		\
			ImGui::NextColumn();	\
		}
#endif

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

#ifdef USE_GRAPHS
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
#ifdef USE_GRAPHS
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
#ifdef USE_GRAPHS
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
		}

#undef SPACE
#undef ROW

#ifdef USE_GRAPHS
#undef USE_GRAPHS
#endif
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
		ImGui::BeginChild("Model list", ImVec2(0, (ImGui::GetWindowHeight() - 100) * .45f));
		if(ImGui::ListBox("Models", &mSelectedModelIx, modelNames.data(), static_cast<int>(modelNames.size())))
			SelectModel(mSelectedModelIx);
		ImGui::EndChild();
		ImGui::NextColumn();

		//free(modelNames);

		// Import model
		if(ImGui::Button("Import"))
		{
			std::string modelFile = ImportModelDialog();
			if(!modelFile.empty())
			{
				std::shared_ptr<Model> model = Importer::ImportModel(GuiHelpers::scene, modelFile);
				if(model)
					GuiHelpers::scene->Add(model);

				models = GuiHelpers::scene->Models();
				mSelectedModelIx = static_cast<int>(models.size() - 1);
				strcpy_s(mModelName, mNameBufferSize, models[mSelectedModelIx]->Name().c_str());
			}
		}

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
		ImGui::BeginChild("Instance list", ImVec2(0, (ImGui::GetWindowHeight() - 100) * .45f));
		if(ImGui::ListBox("Instances", &mSelectedInstanceIx, instanceNames.data(), static_cast<int>(instanceNames.size())))
			SelectInstance(mSelectedInstanceIx);
		ImGui::EndChild();
		ImGui::NextColumn();

		if(instances.size() > 0)
		{
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
