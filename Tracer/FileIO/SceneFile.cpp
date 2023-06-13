#include "SceneFile.h"

// Project
#include "FileIO/JsonHelpers.h"
#include "FileIO/ModelFile.h"
#include "FileIO/TextureFile.h"
#include "OpenGL/Shader.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Renderer/Sky.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Utility/FileSystem.h"
#include "Utility/Logger.h"
#include "Utility/Stopwatch.h"
#include "Utility/Strings.h"

// C++
#include <fstream>
#include <map>
#include <set>
#include <streambuf>
#include <string_view>

using namespace rapidjson;

// Keys
#define Key_AngularDiameter    "angulardiameter"
#define Key_Aperture           "aperture"
#define Key_AoDist             "aodist"
#define Key_Bias               "bias"
#define Key_Camera             "camera"
#define Key_Dir                "dir"
#define Key_Distortion         "distortion"
#define Key_DrawSun            "drawsun"
#define Key_Enabled            "enabled"
#define Key_FocalDist          "focaldist"
#define Key_Fov                "fov"
#define Key_FragPath           "frag"
#define Key_Instances          "instances"
#define Key_Intensity          "intensity"
#define Key_Map                "Map"
#define Key_Materials          "materials"
#define Key_Matrix             "matrix"
#define Key_MaxDepth           "maxdepth"
#define Key_Model              "model"
#define Key_Models             "models"
#define Key_MultiSample        "multisample"
#define Key_Name               "name"
#define Key_Path               "path"
#define Key_Position           "position"
#define Key_Post               "post"
#define Key_RayEpsilon         "ray-epsilon"
#define Key_Renderer           "renderer"
#define Key_RotateX            "rotate-x"
#define Key_RotateY            "rotate-y"
#define Key_RotateZ            "rotate-z"
#define Key_Scale              "scale"
#define Key_Sky                "sky"
#define Key_Target             "target"
#define Key_Transform          "transform"
#define Key_Translate          "translate"
#define Key_Turbidity          "turbidity"
#define Key_Uniforms           "uniforms"
#define Key_Up                 "up"
#define Key_ZDepthMax          "zdepthmax"

namespace Tracer
{
	namespace
	{
		//----------------------------------------------------------------------------------------------------------------------
		// Import sections
		//----------------------------------------------------------------------------------------------------------------------
		void ParseCamera(CameraNode* camNode, const Document& doc)
		{
			if(!camNode || !doc.HasMember(Key_Camera))
				return;

			const Value& jsonCamera = doc[Key_Camera];

			float3 pos = make_float3(0, 0, -1);
			float3 target = make_float3(0, 0, 0);
			float3 up = make_float3(0, 1, 0);
			float aperture = 0;
			float distortion = 0;
			float focalDist = 1e5f;
			float fov = 90.f;

			Read(jsonCamera, Key_Position, pos);
			Read(jsonCamera, Key_Target, target);
			Read(jsonCamera, Key_Up, up);
			Read(jsonCamera, Key_Aperture, aperture);
			Read(jsonCamera, Key_Distortion, distortion);
			Read(jsonCamera, Key_FocalDist, focalDist);
			Read(jsonCamera, Key_Fov, fov);

			// corrections
			fov = fminf(fmaxf(fov, .1f), 179.9f);

			if(pos == target)
				target = pos + make_float3(0, 0, 1);

			// set camera
			*camNode = CameraNode(pos, target, up, fov * DegToRad);
			camNode->SetAperture(aperture);
			camNode->SetDistortion(distortion);
			camNode->SetFocalDist(focalDist);
		}



		struct InstanceInfo
		{
			std::string name = "";
			std::string modelName = "";
			float3x4 transform = make_float3x4();
		};



		std::vector<InstanceInfo> ParseInstances(const Document& doc)
		{
			std::vector<InstanceInfo> importedInstances;

			if(!doc.HasMember(Key_Instances))
				return importedInstances;

			const Value& jsonInstanceList = doc[Key_Instances];
			if(!jsonInstanceList.IsArray())
				return importedInstances;

			for(SizeType instanceIx = 0; instanceIx < jsonInstanceList.Size(); instanceIx++)
			{
				const Value& jsonInstance = jsonInstanceList[instanceIx];

				std::string modelName;
				if(!Read(jsonInstance, Key_Model, modelName))
					continue;

				std::string name = modelName;
				Read(jsonInstance, Key_Name, name);

				float3x4 trans = make_float3x4();
				if(jsonInstance.HasMember(Key_Transform))
				{
					// reusable variables for JSON reading
					float f;
					float3 f3;
					float3x4 f3x4;

					const Value& jsonTransformOperations = jsonInstance[Key_Transform];
					if(jsonTransformOperations.IsArray())
					{
						for(SizeType transOpIx = 0; transOpIx < jsonTransformOperations.Size(); transOpIx++)
						{
							const Value& op = jsonTransformOperations[transOpIx];
							if(Read(op, Key_Matrix, f3x4))
								trans *= f3x4;
							else if(Read(op, Key_Translate, f3))
								trans *= translate_3x4(f3);
							else if(Read(op, Key_RotateX, f))
								trans *= rotate_x_3x4(f * DegToRad);
							else if(Read(op, Key_RotateY, f))
								trans *= rotate_y_3x4(f * DegToRad);
							else if(Read(op, Key_RotateZ, f))
								trans *= rotate_z_3x4(f * DegToRad);
							else if(Read(op, Key_Scale, f))
								trans *= scale_3x4(f);
							else if(Read(op, Key_Scale, f3))
								trans *= scale_3x4(f3);
						}
					}
				}

				importedInstances.push_back({name, modelName, trans});
			}

			return importedInstances;
		}



		void ParseMaterials(Scene* scene, const Document& doc)
		{
			// #TODO: complete implementation

			if(!scene || !doc.HasMember(Key_Materials))
				return;

			const Value& jsonModelList = doc[Key_Materials];
			if(!jsonModelList.IsArray())
				return;

			// for each model
			for(SizeType modelIx = 0; modelIx < jsonModelList.Size(); modelIx++)
			{
				const Value& jsonModel = jsonModelList[modelIx];

				std::string modelName;
				if(!Read(jsonModel, Key_Model, modelName))
					continue;

				std::shared_ptr<Model> model = scene->GetModel(modelName);
				if(!model)
					continue;

				// for each material
				const Value& jsonMaterialList = jsonModel[Key_Materials];
				for(SizeType matIx = 0; matIx < jsonMaterialList.Size(); matIx++)
				{
					const Value& jsonMat = jsonMaterialList[matIx];

					std::string matName;
					if(!Read(jsonMat, Key_Name, matName))
						continue;

					std::shared_ptr<Material> mat = model->GetMaterial(matName);
					if(!mat)
						continue;

					for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
					{
						const MaterialPropertyIds id = static_cast<MaterialPropertyIds>(i);
						const std::string propName = ToLower(std::string(magic_enum::enum_name(id)));

						float f;
						if(Read(jsonMat, propName.c_str(), f))
							mat->Set(id, f);

						float3 f3;
						if(Read(jsonMat, propName.c_str(), f3))
							mat->Set(id, f3);

						std::string s;
						if(Read(jsonMat, (propName + Key_Map).c_str(), s))
							mat->Set(id, s.empty() ? nullptr : TextureFile::Import(scene, s));
					}
				}
			}
		}



		std::map<std::string, std::string> ParseModels(const Document& doc)
		{
			std::map<std::string, std::string> importedModels;

			if(!doc.HasMember(Key_Models))
				return importedModels;

			const Value& jsonModelList = doc[Key_Models];
			if(!jsonModelList.IsArray())
				return importedModels;

			for(SizeType modelIx = 0; modelIx < jsonModelList.Size(); modelIx++)
			{
				const Value& jsonModel = jsonModelList[modelIx];

				std::string name;
				if(!Read(jsonModel, Key_Name, name))
					continue;

				std::string path;
				if(!Read(jsonModel, Key_Path, path))
					continue;

				importedModels[name] = path;
			}

			return importedModels;
		}



		void ParsePostSettings(Window* window, const Document& doc)
		{
			if(!window || !doc.HasMember(Key_Post))
				return;

			const Value& jsonPost = doc[Key_Post];
			if(!jsonPost.IsArray())
				return;

			std::vector<std::shared_ptr<Shader>> shaders;
			for(SizeType shaderIx = 0; shaderIx < jsonPost.Size(); shaderIx++)
			{
				const Value& jsonShader = jsonPost[shaderIx];

				// parse shader
				std::string name;
				if(!Read(jsonShader, Key_Name, name))
					continue;

				std::string fragPath;
				if(!Read(jsonShader, Key_FragPath, fragPath))
					continue;

				// create shader
				std::shared_ptr<Shader> s = std::make_shared<Shader>(name, fragPath, Shader::SourceType::File);

				// parse uniforms
				int i;
				float f;

				const Value& jsonUniforms = jsonShader[Key_Uniforms];
				std::map<std::string, Shader::Uniform>& uniforms = s->Uniforms();
				for (auto& [identifier, uniform] : uniforms)
				{
					switch (uniform.Type())
					{
					case Shader::Uniform::Types::Float:
						if(Read(jsonUniforms, identifier, f))
							uniform.Set(f);
						break;

					case Shader::Uniform::Types::Int:
						if(Read(jsonUniforms, identifier, i))
							uniform.Set(i);
						break;

					case Shader::Uniform::Types::Texture:
						// #TODO: Parse texture paths

					case Shader::Uniform::Types::Unknown:
					default:
						break;
					}
				}
			}

			if(shaders.size() == 0)
				shaders.push_back(std::make_shared<Shader>("Tone Mapping", "glsl/Tonemap.frag", Shader::SourceType::File));

			window->SetPostStack(shaders);
		}



		void ParseRenderSettings(Renderer* renderer, const Document& doc)
		{
			if(!renderer || !doc.HasMember(Key_Renderer))
				return;

			// reusable variables for JSON reading
			int i;
			float f;

			KernelSettings settings = renderer->Settings();
			const Value& jsonRenderer = doc[Key_Renderer];
			if(Read(jsonRenderer, Key_MultiSample, i))
				settings.multiSample = i;
			if(Read(jsonRenderer, Key_MaxDepth, i))
				settings.maxDepth = i;
			if(Read(jsonRenderer, Key_AoDist, f))
				settings.aoDist = f;
			if(Read(jsonRenderer, Key_ZDepthMax, f))
				settings.zDepthMax = f;
			if(Read(jsonRenderer, Key_RayEpsilon, f))
				settings.rayEpsilon = f;
			renderer->SetSettings(settings);
		}



		void ParseSkySettings(Sky* sky, const Document& doc)
		{
			if(!sky || !doc.HasMember(Key_Sky))
				return;

			// reusable variables for JSON reading
			bool b;
			float f;
			float3 f3;

			const Value& jsonSky = doc[Key_Sky];
			if(Read(jsonSky, Key_Enabled, b))
				sky->SetEnabled(b);
			if(Read(jsonSky, Key_DrawSun, b))
				sky->SetDrawSun(b);
			if(Read(jsonSky, Key_Dir, f3))
				sky->SetSunDir(f3);
			if(Read(jsonSky, Key_AngularDiameter, f))
				sky->SetSunAngularDiameter(f);
			if(Read(jsonSky, Key_Intensity, f))
				sky->SetSunIntensity(f);
			if(Read(jsonSky, Key_Turbidity, f))
				sky->SetTurbidity(f);
			if(Read(jsonSky, Key_Bias, f))
				sky->SetSelectionBias(f);
		}



		void ImportModels(Scene* scene, const std::map<std::string, std::string>& models, const std::vector<InstanceInfo>& instances)
		{
			// determine which models to import
			std::set<std::string> modelsToImport;
			for(const InstanceInfo& i : instances)
			{
				if(models.find(i.modelName) != models.end())
					modelsToImport.insert(i.modelName);
			}

			// import models
			std::map<std::string, std::shared_ptr<Model>> importedModels;
			for(const std::string& m : modelsToImport)
			{
				std::shared_ptr<Model> model = ModelFile::Import(scene, models.at(m), m);
				if(model)
				{
					scene->Add(model);
					importedModels[m] = model;
				}
			}

			// create instances
			for(const InstanceInfo& i : instances)
			{
				if(models.find(i.modelName) == models.end())
					Logger::Error("Model \"%s\" was not declared", i.modelName.c_str());
				else if(importedModels.find(i.modelName) != importedModels.end())
					scene->Add(std::make_shared<Instance>(i.name, importedModels[i.modelName], i.transform));
			}
		}



		//----------------------------------------------------------------------------------------------------------------------
		// Export sections
		//----------------------------------------------------------------------------------------------------------------------
		void ExportCamera(CameraNode* camNode, Document& doc, Document::AllocatorType& allocator)
		{
			if(!camNode)
				return;

			Value jsonCam = Value(kObjectType);
			Write(jsonCam, allocator, Key_Position, camNode->Position());
			Write(jsonCam, allocator, Key_Target, camNode->Target());
			Write(jsonCam, allocator, Key_Up, camNode->Up());
			Write(jsonCam, allocator, Key_Aperture, camNode->Aperture());
			Write(jsonCam, allocator, Key_Distortion, camNode->Distortion());
			Write(jsonCam, allocator, Key_FocalDist, camNode->FocalDist());
			Write(jsonCam, allocator, Key_Fov, camNode->Fov() * RadToDeg);

			// add new JSON node to the document
			doc.AddMember(Key_Camera, jsonCam, allocator);
		}



		void ExportInstances(Scene* scene, std::map<std::shared_ptr<Model>, std::string> uniqueModelNames, Document& doc, Document::AllocatorType& allocator)
		{
			if(!scene)
				return;

			const std::vector<std::shared_ptr<Instance>>& instances = scene->Instances();
			if(instances.size() == 0)
				return;

			Value jsonInstanceList(kArrayType);
			for(const std::shared_ptr<Instance>& inst : instances)
			{
				if(!inst->GetModel())
					continue;

				// instance
				Value jsonInst = Value(kObjectType);
				Write(jsonInst, allocator, Key_Name, inst->Name());
				Write(jsonInst, allocator, Key_Model, uniqueModelNames[inst->GetModel()]);
				Value jsonMatrix = Value(kObjectType);
				Write(jsonMatrix, allocator, Key_Matrix, inst->Transform());
				Value jsonTransform = Value(kArrayType);
				jsonTransform.PushBack(jsonMatrix, allocator);
				jsonInst.AddMember(Key_Transform, jsonTransform, allocator);

				// add model to array
				jsonInstanceList.PushBack(jsonInst, allocator);
			}

			// add new JSON node to the document
			doc.AddMember(Key_Instances, jsonInstanceList, allocator);
		}



		void ExportMaterials(Scene* scene, std::map<std::shared_ptr<Model>, std::string> uniqueModelNames, Document& doc, Document::AllocatorType& allocator)
		{
			if(!scene)
				return;

			const std::vector<std::shared_ptr<Model>>& models = scene->Models();
			if(models.size() == 0)
				return;

			Value jsonModelList(kArrayType);
			for(const std::shared_ptr<Model>& model : models)
			{
				Value jsonMaterialList = Value(kArrayType);

				for(const std::shared_ptr<Material>& mat : model->Materials())
				{
					Value jsonMat = Value(kObjectType);

					// name
					Write(jsonMat, allocator, Key_Name, mat->Name());

					// properties
					for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
					{
						const MaterialPropertyIds id = static_cast<MaterialPropertyIds>(i);
						const std::string propName = ToLower(std::string(magic_enum::enum_name(id)));

						if(mat->IsFloatColorEnabled(id))
							Write(jsonMat, allocator, propName, mat->FloatColor(id));

						if(mat->IsRgbColorEnabled(id))
							Write(jsonMat, allocator, propName, mat->RgbColor(id));

						if(mat->IsTextureEnabled(id))
							Write(jsonMat, allocator, propName + Key_Map,
								  mat->TextureMap(id) ? NormalizePath(RelativeFilePath(mat->TextureMap(id)->Path())) : "");
					}

					// add mat to model material array
					jsonMaterialList.PushBack(jsonMat, allocator);
				}

				// add model to array
				Value jsonModel = Value(kObjectType);
				Write(jsonModel, allocator, Key_Model, uniqueModelNames[model]);
				jsonModel.AddMember(Key_Materials, jsonMaterialList, allocator);
				jsonModelList.PushBack(jsonModel, allocator);
			}

			// add new JSON node to the document
			doc.AddMember(Key_Materials, jsonModelList, allocator);
		}



		void ExportModels(std::map<std::shared_ptr<Model>, std::string> uniqueModelNames, Document& doc, Document::AllocatorType& allocator)
		{
			Value jsonModelList(kArrayType);
			for(const auto& [model, name] : uniqueModelNames)
			{
				Value jsonModel = Value(kObjectType);
				Write(jsonModel, allocator, Key_Name, name);
				Write(jsonModel, allocator, Key_Path, NormalizePath(RelativeFilePath(model->FilePath())));

				// add model to array
				jsonModelList.PushBack(jsonModel, allocator);
			}

			// add new JSON node to the document
			doc.AddMember(Key_Models, jsonModelList, allocator);
		}



		void ExportRenderSettings(Renderer*  renderer, Document& doc, Document::AllocatorType& allocator)
		{
			if(!renderer)
				return;

			Value jsonRenderer = Value(kObjectType);
			const KernelSettings& settings = renderer->Settings();
			Write(jsonRenderer, allocator, Key_MultiSample, settings.multiSample);
			Write(jsonRenderer, allocator, Key_MaxDepth, settings.maxDepth);
			Write(jsonRenderer, allocator, Key_AoDist, settings.aoDist);
			Write(jsonRenderer, allocator, Key_ZDepthMax, settings.zDepthMax);
			Write(jsonRenderer, allocator, Key_RayEpsilon, settings.rayEpsilon);

			// add new JSON node to the document
			doc.AddMember(Key_Renderer, jsonRenderer, allocator);
		}



		void ExportSkySettings(Sky* sky, Document& doc, Document::AllocatorType& allocator)
		{
			if(!sky)
				return;

			Value jsonRenderer = Value(kObjectType);
			Write(jsonRenderer, allocator, Key_Enabled, sky->Enabled());
			Write(jsonRenderer, allocator, Key_DrawSun, sky->DrawSun());
			Write(jsonRenderer, allocator, Key_Dir, sky->SunDir());
			Write(jsonRenderer, allocator, Key_AngularDiameter, sky->SunAngularDiameter());
			Write(jsonRenderer, allocator, Key_Intensity, sky->SunIntensity());
			Write(jsonRenderer, allocator, Key_Turbidity, sky->Turbidity());
			Write(jsonRenderer, allocator, Key_Bias, sky->SelectionBias());

			// add new JSON node to the document
			doc.AddMember(Key_Sky, jsonRenderer, allocator);
		}



		void ExportPostSettings(Window* window, Document& doc, Document::AllocatorType& allocator)
		{
			if(!window)
				return;

			Value jsonPost = Value(kArrayType);

			// export shaders
			std::vector<std::shared_ptr<Shader>>& postStack = window->PostStack();
			for (std::shared_ptr<Shader>& shader : postStack)
			{
				Value jsonShader = Value(kObjectType);
				Write(jsonShader, allocator, Key_Name, shader->Name());
				Write(jsonShader, allocator, Key_FragPath, shader->FragmentFile());

				// export uniforms
				Value jsonUniforms = Value(kObjectType);

				int i;
				float f;

				std::map<std::string, Shader::Uniform>& uniforms = shader->Uniforms();
				for (auto& [identifier, uniform] : uniforms)
				{
					switch (uniform.Type())
					{
					case Shader::Uniform::Types::Float:
						uniform.Get(&f);
						Write(jsonUniforms, allocator, identifier, f);
						break;

					case Shader::Uniform::Types::Int:
						uniform.Get(&i);
						Write(jsonUniforms, allocator, identifier, i);
						break;

					case Shader::Uniform::Types::Texture:
						// #TODO: Export texture paths

					case Shader::Uniform::Types::Unknown:
					default:
						break;
					}
				}

				jsonShader.AddMember(Key_Uniforms, jsonUniforms, allocator);
				jsonPost.PushBack(jsonShader, allocator);
			}

			// add new JSON node to the document
			doc.AddMember(Key_Post, jsonPost, allocator);
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// SceneFile
	//--------------------------------------------------------------------------------------------------------------------------
	bool SceneFile::Load(const std::string& sceneFile, Scene* scene, Sky* sky, CameraNode* camNode, Renderer* renderer, Window* window)
	{
		Logger::Info("Loading scene from \"%s\"", sceneFile.c_str());
		Stopwatch sw;

		// read file from disk
		Document doc;
		{
			std::ifstream f(sceneFile);
			if(!f.is_open())
			{
				Logger::Error("Failed to open \"%s\" for reading.", sceneFile.c_str());
				return false;
			}

			IStreamWrapper isw(f);
			doc.ParseStream(isw);
			f.close();
			if(!doc.IsObject())
			{
				Logger::Error("\"%s\" does not contain valid json code.", sceneFile.c_str());
				return false;
			}
		}

		if(scene)
		{
			const std::map<std::string, std::string> models = ParseModels(doc);
			const std::vector<InstanceInfo> instances = ParseInstances(doc);

			ImportModels(scene, models, instances);
			ParseMaterials(scene, doc);
		}

		if(camNode)
			ParseCamera(camNode, doc);

		if(renderer)
			ParseRenderSettings(renderer, doc);

		if(sky)
			ParseSkySettings(sky, doc);

		if(window)
			ParsePostSettings(window, doc);

		Logger::Info("Loaded scenefile in %s", sw.ElapsedString().c_str());
		return true;
	}



	bool SceneFile::Save(const std::string& sceneFile, Scene* scene, Sky* sky, CameraNode* camNode, Renderer* renderer, Window* window)
	{
		std::string globalPath = GlobalPath(sceneFile);
		if(ToLower(FileExtension(globalPath)) != ".json")
			globalPath += ".json";

		Logger::Info("Saving scene to \"%s\"", globalPath.c_str());
		Stopwatch sw;

		// create json document
		Document doc;
		Document::AllocatorType& allocator = doc.GetAllocator();
		doc.SetObject();

		if(scene)
		{
			// gather model names
			std::map<std::shared_ptr<Model>, std::string> uniqueModelNames;
			for(const std::shared_ptr<Model>& m : scene->Models())
			{
				std::string name = m->Name();
				int i = 0;
				while(std::find_if(uniqueModelNames.begin(), uniqueModelNames.end(), [&name](const std::pair<std::shared_ptr<Model>, std::string>& it){ return name == it.second; }) != uniqueModelNames.end())
				{
					name = format("%s_%i", m->Name().c_str(), ++i);
				}

				uniqueModelNames[m] = name;
			}

			// export to JSON
			ExportModels(uniqueModelNames, doc, allocator);
			ExportInstances(scene, uniqueModelNames, doc, allocator);
			ExportMaterials(scene, uniqueModelNames, doc, allocator);
		}

		if(camNode)
			ExportCamera(camNode, doc, allocator);

		if(renderer)
			ExportRenderSettings(renderer, doc, allocator);

		if(sky)
			ExportSkySettings(sky, doc, allocator);

		if(window)
			ExportPostSettings(window, doc, allocator);

		// write to disk
		std::ofstream f(globalPath);
		if(!f.is_open())
		{
			Logger::Error("Failed to open \"%s\" for writing.", sceneFile.c_str());
			return false;
		}

		OStreamWrapper osw(f);
		PrettyWriter<OStreamWrapper> writer(osw);
		writer.SetFormatOptions(PrettyFormatOptions::kFormatSingleLineArray);
		if(!doc.Accept(writer))
		{
			Logger::Error("Failed to write scene to \"%s\".", sceneFile.c_str());
			return false;
		}

		Logger::Info("Saved scenefile in %s", sw.ElapsedString().c_str());
		return true;
	}
}
