#include "SceneGui.h"

// Project
#include "FileIO/Importer.h"
#include "FileIO/SceneFile.h"
#include "GUI/GuiHelpers.h"
#include "Renderer/Scene.h"
#include "Resources/Instance.h"
#include "Resources/Model.h"
#include "Utility/Utility.h"

// ImGUI
#include "imgui/imgui.h"

namespace Tracer
{
	namespace
	{
		std::string ImportModelDialog()
		{
			std::string filter = "";
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
	}



	SceneGui* const SceneGui::Get()
	{
		static SceneGui inst;
		return &inst;
	}



	void SceneGui::DrawImpl()
	{
		ImGui::Begin("Scene", &mEnabled);

		DrawScene();
		ImGui::Spacing();
		ImGui::Spacing();

		DrawModels();
		ImGui::Spacing();
		ImGui::Spacing();

		DrawInstances();
		ImGui::End();
	}



	void SceneGui::DrawScene()
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



	void SceneGui::DrawModels()
	{
		ImGui::Columns(2, nullptr, true);

		// Gather model names
		auto models = GuiHelpers::scene->Models();
		std::vector<const char*> modelNames;
		modelNames.reserve(models.size());
		for(auto m : models)
			modelNames.push_back(m->Name().c_str());

		// Model selection
		ImGui::BeginChild("Model list", ImVec2(0, ImGui::GetWindowHeight() * .4f));
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



	void SceneGui::DrawInstances()
	{
		ImGui::Columns(2, nullptr, true);

		// Gather model names
		auto instances = GuiHelpers::scene->Instances();
		std::vector<const char*> instanceNames;
		instanceNames.reserve(instances.size());
		for(auto m : instances)
			instanceNames.push_back(m->Name().c_str());

		// Instance selection
		ImGui::BeginChild("Instance list", ImVec2(0, ImGui::GetWindowHeight() * .4f));
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



	void SceneGui::SelectModel(int ix)
	{
		mSelectedModelIx = ix;
		const auto& models = GuiHelpers::scene->Models();
		if(static_cast<int>(models.size()) > mSelectedModelIx)
			strcpy_s(mModelName, mNameBufferSize, models[mSelectedModelIx]->Name().c_str());
		else
			mModelName[0] = '\0';
	}



	void SceneGui::SelectInstance(int ix)
	{
		mSelectedInstanceIx = ix;
		const auto& instances = GuiHelpers::scene->Instances();
		if(static_cast<int>(instances.size()) > mSelectedInstanceIx)
			strcpy_s(mInstanceName, mNameBufferSize, instances[mSelectedInstanceIx]->Name().c_str());
		else
			mInstanceName[0] = '\0';
	}
}
