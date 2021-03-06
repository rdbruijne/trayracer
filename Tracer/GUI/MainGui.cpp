#include "MainGui.h"

// Project
#include "FileIO/ModelFile.h"
#include "FileIO/SceneFile.h"
#include "FileIO/TextureFile.h"
#include "GUI/GuiExtensions.h"
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
				for(const FileInfo& f : info)
				{
					std::vector<std::string> extParts = Split(f.ext, ',');
					for(const std::string& e : extParts)
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
				for(const FileInfo& f : info)
				{
					std::vector<std::string> extParts = Split(f.ext, ',');
					for(const std::string& e : extParts)
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



	void MainGui::SelectMaterial(std::weak_ptr<Material> material)
	{
		// check if the material is already selected
		if(!material.owner_before(mSelectedMaterial) && !mSelectedMaterial.owner_before(material))
			return;

		// delete GlTextures for old material
		if(!mSelectedMaterial.expired())
		{
			std::shared_ptr<Material> mat = mSelectedMaterial.lock();
			for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
			{
				const MaterialPropertyIds id = static_cast<MaterialPropertyIds>(i);
				std::shared_ptr<Texture> tex = mat->TextureMap(id);
				if(tex)
					tex->DestroyGLTex();
			}
		}

		// assign new material
		mSelectedMaterial = material;
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
		CameraNode* cameraNode = GuiHelpers::GetCamNode();
		if(!cameraNode)
		{
			ImGui::Text("No camera node detected");
			return;
		}

		ImGui::BeginGroup();

		// transformation
		ImGui::Text("Transformation");

		float pos[] = {cameraNode->Position().x, cameraNode->Position().y, cameraNode->Position().z};
		if(ImGui::InputFloat3("Position", pos))
			cameraNode->SetPosition(make_float3(pos[0], pos[1], pos[2]));

		float target[] = {cameraNode->Target().x, cameraNode->Target().y, cameraNode->Target().z};
		if(ImGui::InputFloat3("Target", target))
			cameraNode->SetTarget(make_float3(target[0], target[1], target[2]));

		float up[] = {cameraNode->Up().x, cameraNode->Up().y, cameraNode->Up().z};
		if(ImGui::InputFloat3("Up", up))
			cameraNode->SetUp(make_float3(up[0], up[1], up[2]));

		ImGui::Spacing();

		// lens
		ImGui::Text("Lens");

		float aperture = cameraNode->Aperture();
		if(ImGui::SliderFloat("Aperture", &aperture, 0.f, 100.f, "%.3f", ImGuiSliderFlags_Logarithmic))
			cameraNode->SetAperture(aperture);

		float distortion = cameraNode->Distortion();
		if(ImGui::SliderFloat("Distortion", &distortion, 0.f, 10.f))
			cameraNode->SetDistortion(distortion);

		float focalDist = cameraNode->FocalDist();
		if(ImGui::SliderFloat("Focal dist", &focalDist, 1.f, 1e6f, "%.3f", ImGuiSliderFlags_Logarithmic))
			cameraNode->SetFocalDist(focalDist);

		float fov = cameraNode->Fov() * RadToDeg;
		if(ImGui::SliderFloat("Fov", &fov, 1.f, 179.f))
			cameraNode->SetFov(fov * DegToRad);

		int bokehSideCount = cameraNode->BokehSideCount();
		if(ImGui::SliderInt("Bokeh side count", &bokehSideCount, 0, 16))
			cameraNode->SetBokehSideCount(bokehSideCount);

		float bokehRotation = cameraNode->BokehRotation() * RadToDeg;
		if(ImGui::SliderFloat("Bokeh rotation", &bokehRotation, 1.f, 179.f))
			cameraNode->SetBokehRotation(bokehRotation * DegToRad);

		ImGui::EndGroup();
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
		for(auto& [key, value] : mDebugItems)
		{
			ImGui::Text(key.c_str());
			ImGui::NextColumn();
			ImGui::Text(value.c_str());
			ImGui::NextColumn();
		}
		ImGui::Columns();
	}



	void MainGui::MaterialElements()
	{
		if(mSelectedMaterial.expired() || mSelectedMaterial.use_count() == 0)
		{
			ImGui::Text("No material selected");
			return;
		}

		// texture resolution
		const ImVec2 textureDisplayRes = ImVec2(100, 100) * ImGui::GetIO().FontGlobalScale;

		// fetch material
		std::shared_ptr<Material> mat = mSelectedMaterial.lock();

		// name
		ImGui::Text(mat->Name().c_str());

		// properties
		for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
		{
			const MaterialPropertyIds id = static_cast<MaterialPropertyIds>(i);
			const std::string propName = std::string(magic_enum::enum_name(id));

			if(ImGui::TreeNode(propName.c_str()))
			{
				// float color
				if(mat->IsFloatColorEnabled(id))
				{
					const std::string colorName = "##" + propName;
					float c = mat->FloatColor(id);
					const float2 cRange = mat->FloatColorRange(id);
					if(ImGui::SliderFloat(colorName.c_str(), &c, cRange.x, cRange.y))
						mat->Set(id, c);
				}

				// rgb color
				if(mat->IsRgbColorEnabled(id))
				{
					const std::string colorName = "##" + propName;
					const ImGuiColorEditFlags colorFlags =
						ImGuiColorEditFlags_HDR |
						ImGuiColorEditFlags_Float |
						ImGuiColorEditFlags_PickerHueWheel;
					float3 c = mat->RgbColor(id);
					if(ImGui::ColorEdit3(colorName.c_str(), reinterpret_cast<float*>(&c), colorFlags))
						mat->Set(id, c);
				}

				// texture
				if(mat->IsTextureEnabled(id))
				{
					const std::string texName = "##" + propName + "map";
					std::shared_ptr<Texture> tex = mat->TextureMap(id);

					if(!tex)
					{
						// load button
						const std::string buttonName = "Load texture" + texName;
						if(ImGui::Button(buttonName.c_str()))
						{
							std::string texFile = ImportTextureDialog();
							if(!texFile.empty())
								mat->Set(id, TextureFile::Import(GuiHelpers::GetScene(), texFile));
						}
					}
					else
					{
						const std::string path = tex->Path();
						const int2 res = tex->Resolution();

						// display texture
						tex->CreateGLTex();
						const size_t texId = static_cast<size_t>(tex->GLTex()->ID());
						if(ImGui::ImageButton(reinterpret_cast<ImTextureID>(texId), textureDisplayRes))
						{
							std::string texFile = ImportTextureDialog();
							if(!texFile.empty())
								mat->Set(id, TextureFile::Import(GuiHelpers::GetScene(), texFile));
						}

						// remove texture button
						const std::string buttonName = "X" + texName;
						ImGui::SameLine();
						if(ImGui::Button(buttonName.c_str()))
							mat->Set(id, nullptr);

						// display name & resolution
						ImGui::Text(format("%s (%i x %i)", path.c_str(), res.x, res.y).c_str());
					}
				}

				ImGui::TreePop();
				ImGui::Separator();
			}
		}
	}



	void MainGui::RendererElements()
	{
		Renderer* renderer = GuiHelpers::GetRenderer();
		if(!renderer)
		{
			ImGui::Text("No renderer node detected");
			return;
		}

		Window* window = GuiHelpers::GetRenderWindow();

		// image
		if(ImGui::Button("Export image"))
		{
			std::string imageFile;
			if(SaveFileDialog("Png\0*.png", "Select an image file", imageFile))
			{
				if(ToLower(FileExtension(imageFile)) != ".png")
					imageFile += ".png";
				renderer->RequestSave(imageFile);
			}
		}

		// resolution
		ImGui::Spacing();
		ImGui::Text("Resolution");

		bool fullscreen = window->IsFullscreen();
		if(ImGui::Checkbox("Fullscreen", &fullscreen))
		{
			window->SetFullscreen(fullscreen);
			mResolution = window->Resolution();
		}

		if(mResolution.x == -1 && mResolution.y == -1)
			mResolution = window->Resolution();

		ImGui::InputInt2("##Resolution", reinterpret_cast<int*>(&mResolution));

		if(ImGui::Button("Apply"))
			window->SetResolution(mResolution);
		ImGui::SameLine();
		if(ImGui::Button("Reset"))
			mResolution = window->Resolution();

		// render mode
		ImGui::Spacing();
		ImGui::Text("Settings");

		RenderModes renderMode = renderer->RenderMode();
		if(ComboBox("Render Mode", renderMode))
			renderer->SetRenderMode(renderMode);

		// kernel settings
		int multiSample = renderer->MultiSample();
		if(ImGui::SliderInt("Multi-sample", &multiSample, 1, Renderer::MaxTraceDepth))
			renderer->SetMultiSample(multiSample);

		int maxDepth = renderer->MaxDepth();
		if(ImGui::SliderInt("Max depth", &maxDepth, 1, 16))
			renderer->SetMaxDepth(maxDepth);

		float aoDist = renderer->AODist();
		if(ImGui::SliderFloat("AO Dist", &aoDist, 0.f, 1e4f, "%.3f", ImGuiSliderFlags_Logarithmic))
			renderer->SetAODist(aoDist);

		float zDepthMax = renderer->ZDepthMax();
		if(ImGui::SliderFloat("Z-Depth max", &zDepthMax, 0.f, 1e4f, "%.3f", ImGuiSliderFlags_Logarithmic))
			renderer->SetZDepthMax(zDepthMax);

		MaterialPropertyIds matPropId = renderer->MaterialPropertyId();
		if(ComboBox("Debug property", matPropId))
			renderer->SetMaterialPropertyId(matPropId);

		// post
		ImGui::Spacing();
		ImGui::Text("Post");

		Window::ShaderProperties shaderProps = window->PostShaderProperties();
		ImGui::SliderFloat("Exposure", &shaderProps.exposure, 0.f, 100.f, "%.3f", ImGuiSliderFlags_Logarithmic);
		ImGui::SliderFloat("Gamma", &shaderProps.gamma, 0.f, 4.f);
		ComboBox("Tonemap method", shaderProps.tonemap);
		window->SetPostShaderProperties(shaderProps);

		// denoiser
		ImGui::Spacing();
		ImGui::Text("Denoiser");

		bool denoising = renderer->DenoisingEnabled();
		if(ImGui::Checkbox("Enabled", &denoising))
			renderer->SetDenoiserEnabled(denoising);

		int32_t denoiserSampleThreshold = renderer->DenoiserSampleThreshold();
		if(ImGui::SliderInt("Sample threshold", &denoiserSampleThreshold, 0, 100))
			renderer->SetDenoiserSampleThreshold(denoiserSampleThreshold);
	}



	void MainGui::SceneElements()
	{
		Scene* scene = GuiHelpers::GetScene();
		if(!scene)
		{
			ImGui::Text("No scene detected");
			return;
		}

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
		std::shared_ptr<Sky> sky = GuiHelpers::GetScene()->GetSky();
		if(!sky)
		{
			ImGui::Text("No sky detected");
			return;
		}

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
		if(ImGui::SliderFloat("Sun size (arc minutes)", &sunSize, 1.f, 10800.f, "%.3f", ImGuiSliderFlags_Logarithmic))
			sky->SetSunAngularDiameter(sunSize);

		float sunIntensity = sky->SunIntensity();
		if(ImGui::SliderFloat("Sun intensity", &sunIntensity, 0.f, 1e6f, "%.3f", ImGuiSliderFlags_Logarithmic))
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

		const Renderer* renderer = GuiHelpers::GetRenderer();
		if(!renderer)
		{
			ImGui::Text("No renderer detected");
			return;
		}

		// fetch stats
		const Scene* scene = GuiHelpers::GetScene();
		const Window* window = GuiHelpers::GetRenderWindow();
		const Renderer::RenderStats renderStats = renderer->Statistics();

		// init column layout
		ImGui::Columns(2);

		// device
		const cudaDeviceProp& devProps = renderer->CudaDeviceProperties();
		ROW("Device", devProps.name);
		SPACE;

		// kernel
		ROW("Kernel", std::string(magic_enum::enum_name(renderer->RenderMode())).c_str());
		ROW("Samples","%d", renderer->SampleCount());
		ROW("Denoised samples","%d", renderer->DenoisedSampleCount());
		ROW("Resolution", "%i x %i", window->Resolution().x, window->Resolution().y);

		SPACE;

		// times
		ROW("FPS", "%.1f", 1e3f / GuiHelpers::GetFrameTimeMs());
		ROW("Frame time", "%.1f ms", GuiHelpers::GetFrameTimeMs());
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
		ROW("Rays", "%.1f M (%.1f M/s)", renderStats.pathCount * 1e-6, PerSec(renderStats.pathCount, GuiHelpers::GetFrameTimeMs()) * 1e-6);
		ROW("Primaries", "%.1f M (%.1f M/s)", renderStats.primaryPathCount * 1e-6, PerSec(renderStats.primaryPathCount, renderStats.primaryPathTimeMs) * 1e-6);
		ROW("Secondaries", "%.1f M (%.1f M/s)", renderStats.secondaryPathCount * 1e-6, PerSec(renderStats.secondaryPathCount, renderStats.secondaryPathTimeMs) * 1e-6);
		ROW("Deep", "%.1f M (%.1f M/s)", renderStats.deepPathCount * 1e-6, PerSec(renderStats.deepPathCount, renderStats.deepPathTimeMs) * 1e-6);
		ROW("Shadow", "%.1f M (%.1f M/s)", renderStats.shadowRayCount * 1e-6, PerSec(renderStats.shadowRayCount, renderStats.shadowTimeMs) * 1e-6);

		SPACE;

		// scene
		ROW("Instance count", "%lld", scene->InstanceCount());
		ROW("Model count", "%lld", scene->InstancedModelCount());
		ROW("Texture count", "%lld", scene->TextureCount());
		ROW("Triangle count", "%s", ThousandSeparators(scene->TriangleCount()).c_str());
		ROW("Unique triangle count", "%s", ThousandSeparators(scene->UniqueTriangleCount()).c_str());
		ROW("Lights", "%s", ThousandSeparators(scene->LightCount()).c_str());
		ROW("Unique lights", "%s", ThousandSeparators(scene->UniqueLightCount()).c_str());

		ImGui::Columns();

#undef SPACE
#undef ROW
	}



	void MainGui::Scene_Scene()
	{
		Scene* scene = GuiHelpers::GetScene();

		constexpr int columns = 4;
		constexpr float buttonWidth = (1.f / columns) * .9f;

		ImGui::Columns(columns, nullptr, false);

		// Load scene
		if(ImGui::Button("Load scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
		{
			std::string sceneFile;
			if(OpenFileDialog("Json\0*.json\0", "Select a scene file", true, sceneFile))
			{
				scene->Clear();
				SceneFile::Load(sceneFile, scene, scene->GetSky().get(), GuiHelpers::GetCamNode(), GuiHelpers::GetRenderer(), GuiHelpers::GetRenderWindow());
			}
		}
		ImGui::NextColumn();

		// Add scene
		if(ImGui::Button("Add scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
		{
			std::string sceneFile;
			if(OpenFileDialog("Json\0*.json\0", "Select a scene file", true, sceneFile))
			{
				SceneFile::Load(sceneFile, scene, nullptr, nullptr, nullptr, nullptr);
			}
		}
		ImGui::NextColumn();

		// Save scene
		if(ImGui::Button("Save scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
		{
			std::string sceneFile;
			if(SaveFileDialog("Json\0*.json\0", "Select a scene file", sceneFile))
				SceneFile::Save(sceneFile, scene, scene->GetSky().get(), GuiHelpers::GetCamNode(), GuiHelpers::GetRenderer(), GuiHelpers::GetRenderWindow());
		}
		ImGui::NextColumn();

		// Clear scene
		if(ImGui::Button("Clear scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
		{
			scene->Clear();
		}

		ImGui::Columns();
	}



	void MainGui::Scene_Models()
	{
		Scene* scene = GuiHelpers::GetScene();

		ImGui::Columns(2, nullptr, true);

		// Gather model names
		std::vector<std::shared_ptr<Model>> models = scene->Models();
		std::vector<const char*> modelNames;
		modelNames.reserve(models.size());
		for(const std::shared_ptr<Model>& m : models)
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
				std::shared_ptr<Model> model = ModelFile::Import(scene, modelFile);
				if(model)
					scene->Add(model);

				models = scene->Models();
				mSelectedModelIx = static_cast<int>(models.size() - 1);
				strcpy_s(mModelName, mNameBufferSize, models[mSelectedModelIx]->Name().c_str());
			}
		}

		if(mSelectedModelIx >= static_cast<int>(models.size()))
			mSelectedModelIx = 0;
		std::shared_ptr<Model> model = models.size() > 0 ? models[mSelectedModelIx] : nullptr;

		// delete
		if(ImGui::Button("Delete##delete_model"))
		{
			scene->Remove(model);
			models = scene->Models();
			SelectModel(0);
		}

		// create instance
		if(ImGui::Button("Create instance") && model)
		{
			scene->Add(std::make_shared<Instance>(model->Name(), model, make_float3x4()));
			strcpy_s(mInstanceName, mNameBufferSize, scene->Instances()[mSelectedInstanceIx]->Name().c_str());
		}

		// Properties
		if(ImGui::InputText("Name##model_name", mModelName, mNameBufferSize, ImGuiInputTextFlags_EnterReturnsTrue) && model && strlen(mModelName) > 0)
			model->SetName(mModelName);

		ImGui::Columns();
	}



	void MainGui::Scene_Instances()
	{
		ImGui::Columns(2, nullptr, true);

		// Gather instance names
		const std::vector<std::shared_ptr<Instance>>& instances = GuiHelpers::GetScene()->Instances();
		std::vector<const char*> instanceNames;
		instanceNames.reserve(instances.size());
		for(std::shared_ptr<Instance> inst : instances)
			instanceNames.push_back(inst->Name().c_str());

		// Instance selection
		if(ImGui::ListBox("Instances", &mSelectedInstanceIx, instanceNames.data(), static_cast<int>(instanceNames.size())))
			SelectInstance(mSelectedInstanceIx);
		ImGui::NextColumn();

		if(instances.size() > 0)
		{
			if(mSelectedInstanceIx >= static_cast<int>(instances.size()))
				mSelectedInstanceIx = 0;

			std::shared_ptr<Instance> inst = instances[mSelectedInstanceIx];
			std::shared_ptr<Model> model = inst->GetModel();

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
			if(ImGui::InputFloat3("Pos", p, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue))
			{
				pos = make_float3(p[0], p[1], p[2]);
				changed = true;
			}

			if(ImGui::InputFloat3("Scale", s, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue))
			{
				scale = make_float3(s[0], s[1], s[2]);
				changed = true;
			}

			if(ImGui::InputFloat3("Euler", e, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue))
			{
				euler = make_float3(e[0] * DegToRad, e[1] * DegToRad, e[2] * DegToRad);
				changed = true;
			}

			if(changed)
				inst->SetTransform(rotate_3x4(euler) * scale_3x4(scale) * translate_3x4(pos));


			// delete
			if(ImGui::Button("Delete##delete_instance"))
			{
				GuiHelpers::GetScene()->Remove(inst);
				SelectInstance(0);
			}
		}

		ImGui::Columns();
	}



	void MainGui::SelectModel(int ix)
	{
		mSelectedModelIx = ix;
		const std::vector<std::shared_ptr<Model>>& models = GuiHelpers::GetScene()->Models();
		if(static_cast<int>(models.size()) > mSelectedModelIx)
			strcpy_s(mModelName, mNameBufferSize, models[mSelectedModelIx]->Name().c_str());
		else
			mModelName[0] = '\0';
	}



	void MainGui::SelectInstance(int ix)
	{
		mSelectedInstanceIx = ix;
		const std::vector<std::shared_ptr<Instance>>& instances = GuiHelpers::GetScene()->Instances();
		if(static_cast<int>(instances.size()) > mSelectedInstanceIx)
			strcpy_s(mInstanceName, mNameBufferSize, instances[mSelectedInstanceIx]->Name().c_str());
		else
			mInstanceName[0] = '\0';
	}
}
