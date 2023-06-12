#include "MainGui.h"

// Project
#include "CUDA/CudaDevice.h"
#include "FileIO/ModelFile.h"
#include "FileIO/SceneFile.h"
#include "FileIO/TextureFile.h"
#include "GUI/GuiExtensions.h"
#include "GUI/GuiHelpers.h"
#include "OpenGL/GLHelpers.h"
#include "OpenGL/Shader.h"
#include "OpenGL/Window.h"
#include "Optix/Denoiser.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Renderer/Sky.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Utility/FileSystem.h"
#include "Utility/Logger.h"
#include "Utility/Strings.h"
#include "Utility/System.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346) // 'name' : dependent name is not a type
#pragma warning(disable: 4626) // 'derived class' : assignment operator was implicitly defined as deleted because a base class assignment operator is inaccessible or deleted
#pragma warning(disable: 5027) // 'type': move assignment operator was implicitly defined as deleted
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// ImGUI
#include "imgui/imgui.h"
#pragma warning(push)
#pragma warning(disable: 4100) // 'identifier' : unreferenced formal parameter
#pragma warning(disable: 4201) // nonstandard extension used : nameless struct/union
#pragma warning(disable: 4263) // 'function' : member function does not override any base class virtual member function
#pragma warning(disable: 4264) // 'virtual_function' : no override available for virtual member function from base 'class'; function is hidden
#pragma warning(disable: 4626) // 'derived class' : assignment operator was implicitly defined as deleted because a base class assignment operator is inaccessible or deleted
#pragma warning(disable: 4458) // declaration of 'identifier' hides class member
#pragma warning(disable: 5027) // 'type': move assignment operator was implicitly defined as deleted
#pragma warning(disable: 5038) // data member 'member1' will be initialized after data member 'member2' | data member 'member' will be initialized after base class 'base_class'
#pragma warning(disable: 5039) // '_function_': pointer or reference to potentially throwing function passed to `extern C` function under `-EHc`. Undefined behavior may occur if this function throws an exception.
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



		inline void TableHeader(const std::string& text)
		{
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::Text(text.c_str());
		}



		template<int Indent=0, typename ... Arg>
		inline void TableRow(const std::string& identifier, const std::string& fmt, Arg... args)
		{
			constexpr float IndentSize = 16.f;

			ImGui::TableNextRow();

			const float indent = IndentSize * (Indent + 1);
			ImGui::TableNextColumn();
			ImGui::Indent(indent);
			ImGui::Text(identifier.c_str());
			ImGui::Unindent(indent);

			ImGui::TableNextColumn();
			ImGui::Text(format(fmt.c_str(), std::forward<Arg>(args)...).c_str());
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
		if(ImGui::Begin("Tray Racer", &mEnabled))
		{
			if(ImGui::CollapsingHeader("Camera"))
				CameraElements();

			if(ImGui::CollapsingHeader("Material"))
				MaterialElements();

			if(ImGui::CollapsingHeader("Post"))
				PostElements();

			if(ImGui::CollapsingHeader("Renderer"))
				RendererElements();

			if(ImGui::CollapsingHeader("Scene"))
				SceneElements();

			if(ImGui::CollapsingHeader("Sky"))
				SkyElements();

			if(ImGui::CollapsingHeader("Statistics"))
				StatisticsElements();

			if(ImGui::CollapsingHeader("Debug"))
				DebugElements();
		}
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
		const ImGuiTableFlags tableFlags = ImGuiTableFlags_RowBg;
		if(ImGui::BeginTable("Debug info", 2, tableFlags))
		{
			for(auto& [key, value] : mDebugItems)
			{
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text(key.c_str());
				ImGui::TableNextColumn();
				ImGui::Text(value.c_str());
			}
			ImGui::EndTable();
		}
	}



	void MainGui::MaterialElements()
	{
		if(mSelectedMaterial.expired() || mSelectedMaterial.use_count() == 0)
		{
			ImGui::Text("No material selected");
			return;
		}

		// fetch material
		std::shared_ptr<Material> mat = mSelectedMaterial.lock();

		const ImGuiTableFlags tableFlags = ImGuiTableFlags_SizingStretchProp;
		if(ImGui::BeginTable(mat->Name().c_str(), 2, tableFlags))
		{
			// header
			ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
			ImGui::TableNextColumn();
			ImGui::TableNextColumn();
			ImGui::Text(mat->Name().c_str());

			const int texScale = 200;
			const ImVec2 textureDisplayRes = ImVec2(texScale, texScale) * ImGui::GetIO().FontGlobalScale;

			// properties
			for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
			{
				const MaterialPropertyIds id = static_cast<MaterialPropertyIds>(i);
				const std::string propName = std::string(magic_enum::enum_name(id));

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				// texture
				if(mat->IsTextureEnabled(id))
				{
					const std::string texName = "##" + propName + "map";
					std::shared_ptr<Texture> tex = mat->TextureMap(id);

					// tooltip
					auto textureTooltip = [tex, textureDisplayRes]() -> void
					{
						if (ImGui::IsItemHovered() && tex.get())
						{
							const std::string path = tex->Path();
							const int2 res = tex->Resolution();

							ImGui::BeginTooltip();
							ImGui::Text(Directory(path).c_str());
							ImGui::Text(FileNameExt(path).c_str());
							ImGui::Text(format("%i x %i", res.x, res.y).c_str());
							tex->CreateGLTex();
							const size_t texId = static_cast<size_t>(tex->GLTex()->ID());
							ImGui::Image(reinterpret_cast<ImTextureID>(texId), textureDisplayRes);
							ImGui::EndTooltip();
						}
					};

					// texture button
					const std::string texButtonLabel = "Texture" + texName;
					if(ImGui::Button(texButtonLabel.c_str()))
					{
						std::string texFile = ImportTextureDialog();
						if(!texFile.empty())
							mat->Set(id, TextureFile::Import(GuiHelpers::GetScene(), texFile));
					}
					textureTooltip();

					// remove texture button
					if(tex)
					{
						const std::string removeButtonLabel = "X" + texName;
						ImGui::SameLine();
						if(ImGui::Button(removeButtonLabel.c_str()))
							mat->Set(id, nullptr);
						textureTooltip();
					}
				}
				ImGui::TableNextColumn();

				// color
				const std::string sliderLabel = propName;
				if(mat->IsFloatColorEnabled(id))
				{
					float c = mat->FloatColor(id);
					const float2 cRange = mat->FloatColorRange(id);
					if(ImGui::SliderFloat(sliderLabel.c_str(), &c, cRange.x, cRange.y))
						mat->Set(id, c);
				}
				else if(mat->IsRgbColorEnabled(id))
				{
					const ImGuiColorEditFlags colorFlags =
						ImGuiColorEditFlags_HDR |
						ImGuiColorEditFlags_Float |
						ImGuiColorEditFlags_PickerHueWheel;
					float3 c = mat->RgbColor(id);
					if(ImGui::ColorEdit3(sliderLabel.c_str(), reinterpret_cast<float*>(&c), colorFlags))
						mat->Set(id, c);
				}
				else
				{
					ImGui::LabelText(sliderLabel.c_str(), "");
				}
			}

			ImGui::EndTable();
		}
	}



	void MainGui::PostElements()
	{
		Window* window = GuiHelpers::GetRenderWindow();
		std::vector<std::shared_ptr<Shader>>& postStack = window->PostStack();

		std::vector<std::shared_ptr<Shader>>::iterator moveUp = postStack.end();
		std::vector<std::shared_ptr<Shader>>::iterator moveDown = postStack.end();
		std::vector<std::shared_ptr<Shader>>::iterator remove = postStack.end();

		int ix = 0;
		for(auto it = postStack.begin(); it != postStack.end(); it++)
		{
			std::shared_ptr<Shader>& shader = *it;
			const std::string label = shader->Name() + "##" + std::to_string(ix++);
			if(ImGui::TreeNode(label.c_str()))
			{
				// move up stack
				ImGui::BeginDisabled(it == postStack.begin());
				if(ImGui::Button("Up"))
					moveUp = it;
				ImGui::EndDisabled();
				ImGui::SameLine();

				// move down stack
				ImGui::BeginDisabled(it == postStack.end() - 1);
				if(ImGui::Button("Down"))
					moveDown = it;
				ImGui::EndDisabled();

				// remove
				ImGui::SameLine();
				if(ImGui::Button("Remove"))
					remove = it;

				// enable/disable
				ImGui::SameLine();
				if(ImGui::Button(shader->IsEnabled() ? "Disable" : "Enable"))
					shader->SetEnabled(!shader->IsEnabled());

				// reload
				ImGui::SameLine();
				if(ImGui::Button("Reload"))
					shader->Compile();

				ImGui::Spacing();

				if(!shader->IsValid())
				{
					ImGui::TextColored(ImVec4(1, 0, 0, 1), "Invalid shader:");
					ImGui::Text(shader->ErrorLog().c_str());
					ImGui::TreePop();
					continue;
				}

				// uniforms
				int i;
				float f;
				std::map<std::string, Shader::Uniform>& uniforms = shader->Uniforms();
				for(auto& [identifier, uniform] : uniforms)
				{
					if(Shader::IsInternalUniform(identifier))
						continue;

					switch(uniform.Type())
					{
					case Shader::Uniform::Types::Float:
						uniform.Get(&f);
						if(uniform.HasRange())
						{
							float min, max;
							uniform.GetRange(&min, &max);
							if(ImGui::SliderFloat(identifier.c_str(), &f, min, max, "%.3f", uniform.IsLogarithmic() ? ImGuiSliderFlags_Logarithmic : 0))
								uniform.Set(f);
						}
						else if(ImGui::InputFloat(identifier.c_str(), &f))
						{
							uniform.Set(f);
						}
						break;

					case Shader::Uniform::Types::Int:
						// #TODO: Enum editor
						uniform.Get(&i);
						if(uniform.IsEnum())
						{
							if(ComboBox(identifier, i, uniform.EnumKeys()))
								uniform.Set(i);
						}
						else if(uniform.HasRange())
						{
							int min, max;
							uniform.GetRange(&min, &max);
							if(ImGui::SliderInt(identifier.c_str(), &i, min, max, "%d", uniform.IsLogarithmic() ? ImGuiSliderFlags_Logarithmic : 0))
								uniform.Set(i);
						}
						else
						{
							if(ImGui::InputInt(identifier.c_str(), &i))
								uniform.Set(i);
						}
						break;

					case Shader::Uniform::Types::Texture:
						{
							ImGui::Text(format("[tex] %s", identifier.c_str()).c_str());
						}
						break;

					case Shader::Uniform::Types::Unknown:
					default:
						break;
					}
				}

				ImGui::Separator();
				ImGui::TreePop();
			}
		}

		// swap shaders
		if(moveUp != postStack.end())
		{
			std::iter_swap(moveUp, moveUp - 1);
			window->SetPostStack(postStack);
		}

		if(moveDown != postStack.end())
		{
			std::iter_swap(moveDown, moveDown + 1);
			window->SetPostStack(postStack);
		}

		// remove shader
		if(remove != postStack.end())
		{
			postStack.erase(remove);
			window->SetPostStack(postStack);
		}

		// add new shader
		if(ImGui::Button("Add"))
		{
			// #TODO: Shader creation window
			std::string fragmentFile;
			if(OpenFileDialog("Frag\0*.frag\0", "Select a fragment file", true, fragmentFile))
			{
				Logger::Debug("Adding new shader:");
				Logger::Debug("Fragment: %s", fragmentFile.c_str());
				postStack.push_back(std::make_shared<Shader>(FileName(fragmentFile), Shader::FullScreenQuadVert(), Shader::SourceType::Code, fragmentFile, Shader::SourceType::File));
				window->SetPostStack(postStack);
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
		KernelSettings settings = renderer->Settings();
		bool settingsChanged = false;
		settingsChanged = ImGui::SliderInt("Multi-sample", &settings.multiSample, 1, Renderer::MaxTraceDepth) || settingsChanged;
		settingsChanged = ImGui::SliderInt("Max depth", &settings.maxDepth, 1, 16) || settingsChanged;
		settingsChanged = ImGui::SliderFloat("Ray epsilon", &settings.rayEpsilon, 0.f, 1.f, "%.5f", ImGuiSliderFlags_Logarithmic) || settingsChanged;

		ImGui::BeginDisabled(renderMode != RenderModes::AmbientOcclusion && renderMode != RenderModes::AmbientOcclusionShading);
		settingsChanged = ImGui::SliderFloat("AO Dist", &settings.aoDist, 0.f, 1e4f, "%.3f", ImGuiSliderFlags_Logarithmic) || settingsChanged;
		ImGui::EndDisabled();

		ImGui::BeginDisabled(renderMode != RenderModes::ZDepth);
		settingsChanged = ImGui::SliderFloat("Z-Depth max", &settings.zDepthMax, 0.f, 1e4f, "%.3f", ImGuiSliderFlags_Logarithmic) || settingsChanged;
		ImGui::EndDisabled();

		if(settingsChanged)
			renderer->SetSettings(settings);

		// debug property
		MaterialPropertyIds matPropId = renderer->MaterialPropertyId();
		ImGui::BeginDisabled(renderMode != RenderModes::MaterialProperty);
		if(ComboBox("Material property", matPropId))
			renderer->SetMaterialPropertyId(matPropId);
		ImGui::EndDisabled();

		// denoiser
		ImGui::Spacing();
		ImGui::Text("Denoiser");

		std::shared_ptr<Denoiser> denoiser = renderer->GetDenoiser();

		bool denoising = denoiser->IsEnabled();
		if(ImGui::Checkbox("Enabled", &denoising))
			denoiser->SetEnabled(denoising);

		int32_t denoiserSampleThreshold = static_cast<int32_t>(denoiser->SampleTreshold());
		if(ImGui::SliderInt("Sample threshold", &denoiserSampleThreshold, 0, 100))
			denoiser->SetSampleTreshold(static_cast<uint32_t>(denoiserSampleThreshold < 0 ? 0 : denoiserSampleThreshold));
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

		float selectionBias = sky->SelectionBias();
		if(ImGui::SliderFloat("Selection Bias", &selectionBias, 0.f, 100.f))
			sky->SetSelectionBias(selectionBias);

		// ground
		ImGui::Text("Ground");

		float turbidity = sky->Turbidity();
		if(ImGui::SliderFloat("Turbidity", &turbidity, 1.f, 10.f))
			sky->SetTurbidity(turbidity);
	}



	void MainGui::StatisticsElements()
	{
		const Renderer* renderer = GuiHelpers::GetRenderer();
		if(!renderer)
		{
			ImGui::Text("No renderer detected");
			return;
		}

		// fetch stats
		const Scene* scene = GuiHelpers::GetScene();
		const Window* window = GuiHelpers::GetRenderWindow();
		const RenderStatistics renderStats = renderer->Statistics();
		const RenderStatistics::DeviceStatistics& deviceStats = renderStats.device;

		// init column layout
		const ImGuiTableFlags tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner;
		if(ImGui::BeginTable("statistics", 2, tableFlags))
		{
			// device
			const cudaDeviceProp& devProps = renderer->Device()->DeviceProperties();
			size_t freeDeviceMem, totalDeviceMem;
			renderer->Device()->MemoryUsage(freeDeviceMem, totalDeviceMem);
			TableHeader("Devices");
			TableRow("Device", devProps.name);
			TableRow("Memory used / total", "%d / %d MB", (totalDeviceMem - freeDeviceMem) >> 20, totalDeviceMem >> 20);

			// kernel
			TableHeader("Kernel");
			TableRow("Kernel", std::string(magic_enum::enum_name(renderer->RenderMode())).c_str());
			TableRow("Samples", "%d", renderer->SampleCount());
			TableRow("Denoised samples","%d", renderer->GetDenoiser()->SampleCount());
			TableRow("Resolution", "%i x %i", window->Resolution().x, window->Resolution().y);

			// times
			TableHeader("Framerate");
			TableRow("FPS", "%.1f", 1e3f / GuiHelpers::GetFrameTimeMs());
			TableRow("Frame time", "%.1f ms", GuiHelpers::GetFrameTimeMs());

			TableHeader("Ray times");
			TableRow("Primary rays", "%.1f ms", deviceStats.primaryPathTimeMs);
			TableRow("Secondary rays", "%.1f ms", deviceStats.secondaryPathTimeMs);
			TableRow("Deep rays", "%.1f ms", deviceStats.deepPathTimeMs);
			TableRow("Shadow rays", "%.1f ms", deviceStats.shadowTimeMs);

			TableHeader("Post");
			TableRow("Shade time", "%.1f ms", deviceStats.shadeTimeMs);
			TableRow("Denoise time", "%.1f ms", renderStats.denoiseTimeMs);

			TableHeader("Build");
			TableRow("Build time", "%.1f ms", renderStats.buildTimeMs);
			TableRow("Geometry build time", "%.1f ms", renderStats.geoBuildTimeMs);
			TableRow("Material build time", "%.1f ms", renderStats.matBuildTimeMs);
			TableRow("Sky build time", "%.1f ms", renderStats.skyBuildTimeMs);

			TableHeader("Upload");
			TableRow("Geometry upload time", "%.1f ms", renderStats.geoUploadTimeMs);
			TableRow("Material upload time", "%.1f ms", renderStats.matUploadTimeMs);
			TableRow("Sky upload time", "%.1f ms", renderStats.skyUploadTimeMs);

			// rays
			TableHeader("Rays/sec");
			TableRow("Rays", "%.1f M (%.1f M/s)", static_cast<float>(deviceStats.pathCount) / 1e6f, PerSec(deviceStats.pathCount, GuiHelpers::GetFrameTimeMs()) / 1e6f);
			TableRow("Primaries", "%.1f M (%.1f M/s)", static_cast<float>(deviceStats.primaryPathCount) / 1e6f, PerSec(deviceStats.primaryPathCount, deviceStats.primaryPathTimeMs) / 1e6f);
			TableRow("Secondaries", "%.1f M (%.1f M/s)", static_cast<float>(deviceStats.secondaryPathCount) / 1e6f, PerSec(deviceStats.secondaryPathCount, deviceStats.secondaryPathTimeMs) / 1e6f);
			TableRow("Deep", "%.1f M (%.1f M/s)", static_cast<float>(deviceStats.deepPathCount) / 1e6f, PerSec(deviceStats.deepPathCount, deviceStats.deepPathTimeMs) / 1e6f);
			TableRow("Shadow", "%.1f M (%.1f M/s)", static_cast<float>(deviceStats.shadowRayCount) / 1e6f, PerSec(deviceStats.shadowRayCount, deviceStats.shadowTimeMs) / 1e6f);

			// scene
			TableHeader("Scene");
			TableRow("Instance count", "%lld", scene->InstanceCount());
			TableRow("Model count", "%lld", scene->InstancedModelCount());
			TableRow("Texture count", "%lld", scene->TextureCount());
			TableRow("Triangle count", "%s", ThousandSeparators(scene->TriangleCount()).c_str());
			TableRow("Unique triangle count", "%s", ThousandSeparators(scene->UniqueTriangleCount()).c_str());
			TableRow("Lights", "%s", ThousandSeparators(scene->LightCount()).c_str());
			TableRow("Unique lights", "%s", ThousandSeparators(scene->UniqueLightCount()).c_str());
			TableRow("Light energy", "%f", scene->LightEnergy());
			TableRow("Sun energy", "%f", scene->SunEnergy());

			// drivers
			int driverVersion;
			int cudaRuntime;
			CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
			CUDA_CHECK(cudaRuntimeGetVersion(&cudaRuntime));
			TableHeader("CUDA");
			TableRow("CUDA API", "%d.%d", CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10);
			TableRow("CUDA driver", "%d.%d", driverVersion / 1000, (driverVersion % 1000) / 10);
			TableRow("CUDA runtime", "%d.%d", cudaRuntime / 1000, (cudaRuntime % 1000) / 10);
			TableRow("Optix", "%d.%d", OPTIX_VERSION / 10000, (OPTIX_VERSION % 10000) / 100, OPTIX_VERSION % 100);

			// OpenGL
			TableHeader("OpenGL");
			TableRow("OpenGL Version", GLVersion());
			TableRow("OpenGL Vendor", GLVendor());
			TableRow("OpenGL Renderer", GLRenderer());
			TableRow("OpenGL Shading Language Version", GLShadingLanguageVersion());

			ImGui::EndTable();
		}
	}



	void MainGui::Scene_Scene()
	{
		Scene* scene = GuiHelpers::GetScene();

		constexpr int columns = 4;
		constexpr float buttonWidth = (1.f / columns) * .9f;
		const ImGuiTableFlags tableFlags = ImGuiTableFlags_None;
		if(ImGui::BeginTable("scene buttons", 4, tableFlags))
		{
			ImGui::TableNextRow();

			// Load scene
			ImGui::TableNextColumn();
			if(ImGui::Button("Load scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
			{
				std::string sceneFile;
				if(OpenFileDialog("Json\0*.json\0", "Select a scene file", true, sceneFile))
				{
					scene->Clear();
					SceneFile::Load(sceneFile, scene, scene->GetSky().get(), GuiHelpers::GetCamNode(), GuiHelpers::GetRenderer(), GuiHelpers::GetRenderWindow());
				}
			}

			// Add scene
			ImGui::TableNextColumn();
			if(ImGui::Button("Add scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
			{
				std::string sceneFile;
				if(OpenFileDialog("Json\0*.json\0", "Select a scene file", true, sceneFile))
				{
					SceneFile::Load(sceneFile, scene, nullptr, nullptr, nullptr, nullptr);
				}
			}

			// Save scene
			ImGui::TableNextColumn();
			if(ImGui::Button("Save scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
			{
				std::string sceneFile;
				if(SaveFileDialog("Json\0*.json\0", "Select a scene file", sceneFile))
					SceneFile::Save(sceneFile, scene, scene->GetSky().get(), GuiHelpers::GetCamNode(), GuiHelpers::GetRenderer(), GuiHelpers::GetRenderWindow());
			}

			// Clear scene
			ImGui::TableNextColumn();
			if(ImGui::Button("Clear scene", ImVec2(ImGui::GetWindowWidth() * buttonWidth, 0)))
			{
				scene->Clear();
			}

			ImGui::EndTable();
		}
	}



	void MainGui::Scene_Models()
	{
		Scene* scene = GuiHelpers::GetScene();

		// Gather model names
		std::vector<std::shared_ptr<Model>> models = scene->Models();
		std::vector<const char*> modelNames;
		modelNames.reserve(models.size());
		for(const std::shared_ptr<Model>& m : models)
			modelNames.push_back(m->Name().c_str());

		const ImGuiTableFlags tableFlags = ImGuiTableFlags_BordersInnerV;
		if(ImGui::BeginTable("Scene models", 2, tableFlags))
		{
			ImGui::TableNextRow();

			// Model selection
			ImGui::TableNextColumn();
			if(ImGui::ListBox("Models", &mSelectedModelIx, modelNames.data(), static_cast<int>(modelNames.size())))
				SelectModel(mSelectedModelIx);

			// model manipulation
			ImGui::TableNextColumn();

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
					strcpy_s(mModelName, mNameBufferSize, models[static_cast<size_t>(mSelectedModelIx)]->Name().c_str());
				}
			}

			if(mSelectedModelIx >= static_cast<int>(models.size()))
				mSelectedModelIx = 0;
			std::shared_ptr<Model> model = models.size() > 0 ? models[static_cast<size_t>(mSelectedModelIx)] : nullptr;

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
				strcpy_s(mInstanceName, mNameBufferSize, scene->Instances()[static_cast<size_t>(mSelectedInstanceIx)]->Name().c_str());
			}

			// Properties
			if(ImGui::InputText("Name##model_name", mModelName, mNameBufferSize, ImGuiInputTextFlags_EnterReturnsTrue) && model && strlen(mModelName) > 0)
				model->SetName(mModelName);

			ImGui::EndTable();
		}
	}



	void MainGui::Scene_Instances()
	{
		// Gather instance names
		const std::vector<std::shared_ptr<Instance>>& instances = GuiHelpers::GetScene()->Instances();
		std::vector<const char*> instanceNames;
		instanceNames.reserve(instances.size());
		for(std::shared_ptr<Instance> inst : instances)
			instanceNames.push_back(inst->Name().c_str());

		const ImGuiTableFlags tableFlags = ImGuiTableFlags_BordersInnerV;
		if(ImGui::BeginTable("Scene instances", 2, tableFlags))
		{
			ImGui::TableNextRow();

			// Instance selection
			ImGui::TableNextColumn();
			if(ImGui::ListBox("Instances", &mSelectedInstanceIx, instanceNames.data(), static_cast<int>(instanceNames.size())))
				SelectInstance(mSelectedInstanceIx);

			// manipulate instance
			ImGui::TableNextColumn();
			if(instances.size() > 0)
			{
				if(mSelectedInstanceIx >= static_cast<int>(instances.size()))
					mSelectedInstanceIx = 0;

				std::shared_ptr<Instance> inst = instances[static_cast<size_t>(mSelectedInstanceIx)];
				std::shared_ptr<Model> model = inst->GetModel();

				// Name
				if(ImGui::InputText("Name##inst_name", mInstanceName, mNameBufferSize, ImGuiInputTextFlags_EnterReturnsTrue) && inst && strlen(mInstanceName) > 0)
					inst->SetName(mInstanceName);

				ImGui::InputText("Model",
								 model ? const_cast<char*>(model->Name().c_str()) : nullptr,
								 model ? static_cast<int>(model->Name().length()) : 0ull,
								 ImGuiInputTextFlags_ReadOnly);

				// transform
				float3 pos;
				float3 scale;
				float3 euler;
				inst->DecomposedTransform(pos, euler, scale);

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
				{
					inst->SetDecomposedTransform(pos, euler, scale);
				}

				// show/hide
				if(ImGui::Button(inst->IsVisible() ? "Hide" : "Show"))
				{
					inst->SetVisible(!inst->IsVisible());
				}

				// delete
				ImGui::SameLine();
				if(ImGui::Button("Delete##delete_instance"))
				{
					GuiHelpers::GetScene()->Remove(inst);
					SelectInstance(0);
				}
			}

			ImGui::EndTable();
		}
	}



	void MainGui::SelectModel(int ix)
	{
		mSelectedModelIx = ix;
		const std::vector<std::shared_ptr<Model>>& models = GuiHelpers::GetScene()->Models();
		if(static_cast<int>(models.size()) > mSelectedModelIx)
			strcpy_s(mModelName, mNameBufferSize, models[static_cast<size_t>(mSelectedModelIx)]->Name().c_str());
		else
			mModelName[0] = '\0';
	}



	void MainGui::SelectInstance(int ix)
	{
		mSelectedInstanceIx = ix;
		const std::vector<std::shared_ptr<Instance>>& instances = GuiHelpers::GetScene()->Instances();
		if(static_cast<int>(instances.size()) > mSelectedInstanceIx)
			strcpy_s(mInstanceName, mNameBufferSize, instances[static_cast<size_t>(mSelectedInstanceIx)]->Name().c_str());
		else
			mInstanceName[0] = '\0';
	}
}
