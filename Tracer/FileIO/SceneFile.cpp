#include "SceneFile.h"

// Project
#include "FileIO/Importer.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Resources/Model.h"

// RapidJson
#pragma warning(push)
#pragma warning(disable: 4061 4464 4619)
#include "RapidJson/document.h"
#include "RapidJson/istreamwrapper.h"
#include "RapidJson/ostreamwrapper.h"
#include "RapidJson/prettywriter.h"
#pragma warning(pop)

// C++
#include <fstream>
#include <map>
#include <set>
#include <streambuf>

// #TODO: clean code by adding helper functions in nameless namespace

using namespace rapidjson;

namespace Tracer
{
	namespace
	{
		//----------------------------------------------------------------------------------------------------------------------
		// read helpers
		//----------------------------------------------------------------------------------------------------------------------
		bool Read(const Value& jsonValue, const char* memberName, float& result)
		{
			if(!jsonValue.HasMember(memberName))
				return false;

			const Value& val = jsonValue[memberName];
			if(!val.IsNumber())
				return false;

			result = val.GetFloat();
			return true;
		}



		bool Read(const Value& jsonValue, const char* memberName, int& result)
		{
			if(!jsonValue.HasMember(memberName))
				return false;

			const Value& val = jsonValue[memberName];
			if(!val.IsNumber())
				return false;

			result = val.GetInt();
			return true;
		}



		bool Read(const Value& jsonValue, const char* memberName, float3& result)
		{
			if(!jsonValue.HasMember(memberName))
				return false;

			const Value& val = jsonValue[memberName];
			if(!val.IsArray() || val.Size() != 3 || !val[0].IsNumber() || !val[1].IsNumber() || !val[2].IsNumber())
				return false;

			result = make_float3(val[0].GetFloat(), val[1].GetFloat(), val[2].GetFloat());
			return true;
		}



		bool Read(const Value& jsonValue, const char* memberName, float3x4& result)
		{
			if(!jsonValue.HasMember(memberName))
				return false;

			const Value& val = jsonValue[memberName];
			if(!val.IsArray() || val.Size() != 12)
				return false;

			float m[12] = {};
			for(SizeType i = 0; i < val.Size(); i++)
			{
				if(!val[i].IsNumber())
					return false;
				m[i] = val[i].GetFloat();
			}

			result = make_float3x4(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]);
			return true;
		}


		bool Read(const Value& jsonValue, const char* memberName, std::string& result)
		{
			if(!jsonValue.HasMember(memberName))
				return false;

			const Value& val = jsonValue[memberName];
			if(!val.IsString())
				return false;

			result = val.GetString();
			return true;
		}



		//----------------------------------------------------------------------------------------------------------------------
		// Import sections
		//----------------------------------------------------------------------------------------------------------------------
		void ParseCamera(CameraNode* camNode, const Document& doc)
		{
			if(!camNode || !doc.HasMember("camera"))
				return;

			const Value& jsonCamera = doc["camera"];

			float3 pos = make_float3(0, 0, -1);
			float3 target = make_float3(0, 0, 0);
			float3 up = make_float3(0, 1, 0);
			float fov = 90.f;

			Read(jsonCamera, "position", pos);
			Read(jsonCamera, "target", target);
			Read(jsonCamera, "up", up);
			Read(jsonCamera, "fov", fov);

			// corrections
			fov = fminf(fmaxf(fov, .1f), 179.9f);

			if(pos == target)
				target = pos + make_float3(0, 0, 1);

			// set camera
			*camNode = CameraNode(pos, target, up, fov * DegToRad);
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

			if(!doc.HasMember("instances"))
				return importedInstances;

			const Value& jsonInstanceList = doc["instances"];
			if(!jsonInstanceList.IsArray())
				return importedInstances;

			for(SizeType instanceIx = 0; instanceIx < jsonInstanceList.Size(); instanceIx++)
			{
				const Value& jsonInstance = jsonInstanceList[instanceIx];

				std::string modelName;
				if(!Read(jsonInstance, "model", modelName))
					continue;

				std::string name = modelName;
				Read(jsonInstance, "name", name);

				float3x4 trans = make_float3x4();
				if(jsonInstance.HasMember("transform"))
				{
					// reusable variables for JSON reading
					float f;
					float3 f3;
					float3x4 f3x4;

					const Value& jsonTransformOperations = jsonInstance["transform"];
					if(jsonTransformOperations.IsArray())
					{
						for(SizeType transOpIx = 0; transOpIx < jsonTransformOperations.Size(); transOpIx++)
						{
							const Value& op = jsonTransformOperations[transOpIx];
							if(Read(op, "matrix", f3x4))
								trans *= f3x4;
							else if(Read(op, "translate", f3))
								trans *= translate_3x4(f3);
							else if(Read(op, "rotate-x", f))
								trans *= rotate_x_3x4(f * DegToRad);
							else if(Read(op, "rotate-y", f))
								trans *= rotate_y_3x4(f * DegToRad);
							else if(Read(op, "rotate-z", f))
								trans *= rotate_z_3x4(f * DegToRad);
							else if(Read(op, "scale", f))
								trans *= scale_3x4(f);
							else if(Read(op, "scale", f3))
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

			if(!scene || !doc.HasMember("materials"))
				return;

			const Value& jsonModelList = doc["materials"];
			if(!jsonModelList.IsArray())
				return;

			// for each model
			for(SizeType modelIx = 0; modelIx < jsonModelList.Size(); modelIx++)
			{
				const Value& jsonModel = jsonModelList[modelIx];

				std::string modelName;
				if(!Read(jsonModel, "model", modelName))
					continue;

				auto model = scene->GetModel(modelName);
				if(!model)
					continue;

				// for each material
				const Value& jsonMaterialList = jsonModel["materials"];
				for(SizeType matIx = 0; matIx < jsonMaterialList.Size(); matIx++)
				{
					const Value& jsonMat = jsonMaterialList[matIx];

					std::string matName;
					if(!Read(jsonMat, "name", matName))
						continue;

					auto mat = model->GetMaterial(matName);
					if(!mat)
						continue;

					// reusable variables for JSON reading
					//float f;
					float3 f3;
					std::string s;

					// properties
					if(Read(jsonMat, "diffuse", f3))
						mat->SetDiffuse(f3);
					if(Read(jsonMat, "emissive", f3))
						mat->SetEmissive(f3);

					// textures
					if(Read(jsonMat, "diffuseMap", s))
						mat->SetDiffuseMap(Importer::ImportTexture(scene, s));
					if(Read(jsonMat, "normalMap", s))
						mat->SetNormalMap(Importer::ImportTexture(scene, s));
				}
			}
		}



		std::map<std::string, std::string> ParseModels(const Document& doc)
		{
			std::map<std::string, std::string> importedModels;

			if(!doc.HasMember("models"))
				return importedModels;

			const Value& jsonModelList = doc["models"];
			if(!jsonModelList.IsArray())
				return importedModels;

			for(SizeType modelIx = 0; modelIx < jsonModelList.Size(); modelIx++)
			{
				const Value& jsonModel = jsonModelList[modelIx];

				std::string name;
				if(!Read(jsonModel, "name", name))
					continue;

				std::string path;
				if(!Read(jsonModel, "path", path))
					continue;

				importedModels[name] = path;
			}

			return importedModels;
		}



		void ParsePostSettings(Window* window, const Document& doc)
		{
			if(!window || !doc.HasMember("post"))
				return;

			Window::ShaderProperties postProps = window->PostShaderProperties();
			const Value& jsonPost = doc["post"];

			Read(jsonPost, "exposure", postProps.exposure);
			Read(jsonPost, "gamma", postProps.gamma);

			window->SetPostShaderProperties(postProps);
		}



		void ParseRenderSettings(Renderer* renderer, const Document& doc)
		{
			if(!renderer || !doc.HasMember("renderer"))
				return;

			// reusable variables for JSON reading
			int i;
			float f;
			float3 f3;

			const Value& jsonRenderer = doc["renderer"];
			if(Read(jsonRenderer, "multisample", i))
				renderer->SetMultiSample(i);
			if(Read(jsonRenderer, "maxdepth", i))
				renderer->SetMaxDepth(i);
			if(Read(jsonRenderer, "aodist", f))
				renderer->SetAODist(f);
			if(Read(jsonRenderer, "zdepthmax", f))
				renderer->SetZDepthMax(f);
			if(Read(jsonRenderer, "skycolor", f3))
				renderer->SetSkyColor(f3);
		}



		void ImportModels(Scene* scene, const std::map<std::string, std::string>& models, const std::vector<InstanceInfo>& instances)
		{
			// determine which models to import
			std::set<std::string> modelsToImport;
			for(auto& i : instances)
			{
				if(models.find(i.modelName) != models.end())
					modelsToImport.insert(i.modelName);
			}

			// import models
			std::map<std::string, std::shared_ptr<Model>> importedModels;
			for(auto m : modelsToImport)
			{
				auto model = Importer::ImportModel(scene, models.at(m), m);
				scene->Add(model);
				importedModels[m] = model;
			}

			// create instances
			for(auto& i : instances)
			{
				if(models.find(i.modelName) != models.end())
					scene->Add(std::make_shared<Instance>(i.name, importedModels[i.modelName], i.transform));
				else
					printf("Model \"%s\" was not declared\n", i.modelName.c_str());
			}
		}



		//----------------------------------------------------------------------------------------------------------------------
		// write helpers
		//----------------------------------------------------------------------------------------------------------------------
		void Write(Value& jsonValue, Document::AllocatorType& allocator, const char* memberName, float val)
		{
			Value v = Value(kObjectType);
			v.SetFloat(val);
			jsonValue.AddMember(StringRef(memberName), v, allocator);
		}



		void Write(Value& jsonValue, Document::AllocatorType& allocator, const char* memberName, int val)
		{
			Value v = Value(kObjectType);
			v.SetInt(val);
			jsonValue.AddMember(StringRef(memberName), v, allocator);
		}



		void Write(Value& jsonValue, Document::AllocatorType& allocator, const char* memberName, float3 val)
		{
			Value jsonVector = Value(kArrayType);
			jsonVector.PushBack(val.x, allocator);
			jsonVector.PushBack(val.y, allocator);
			jsonVector.PushBack(val.z, allocator);

			jsonValue.AddMember(StringRef(memberName), jsonVector, allocator);
		}



		void Write(Value& jsonValue, Document::AllocatorType& allocator, const char* memberName, float3x4 val)
		{
			const float tempVals[12] =
			{
				val.x.x, val.x.y, val.x.z,
				val.y.x, val.y.y, val.y.z,
				val.z.x, val.z.y, val.z.z,
				val.tx, val.ty, val.tz
			};

			Value jsonMatrix = Value(kArrayType);
			for(int i = 0; i < 12; i++)
			{
				Value f = Value(kObjectType);
				f.SetFloat(tempVals[i]);
				jsonMatrix.PushBack(f, allocator);
			}

			jsonValue.AddMember(StringRef(memberName), jsonMatrix, allocator);
		}



		void Write(Value& jsonValue, Document::AllocatorType& allocator, const char* memberName, const std::string& val)
		{
			Value v = Value(kObjectType);
			v.SetString(val.c_str(), static_cast<SizeType>(val.length()), allocator);
			jsonValue.AddMember(StringRef(memberName), v, allocator);
		}



		//----------------------------------------------------------------------------------------------------------------------
		// Export sections
		//----------------------------------------------------------------------------------------------------------------------
		void ExportCamera(CameraNode* camNode, Document& doc, Document::AllocatorType& allocator)
		{
			if(!camNode)
				return;

			Value jsonCam = Value(kObjectType);
			Write(jsonCam, allocator, "position", camNode->Position());
			Write(jsonCam, allocator, "target", camNode->Target());
			Write(jsonCam, allocator, "up", camNode->Up());
			Write(jsonCam, allocator, "fov", camNode->Fov() * RadToDeg);

			// add new JSON node to the document
			doc.AddMember("camera", jsonCam, allocator);
		}



		void ExportInstances(Scene* scene, std::map<std::shared_ptr<Model>, std::string> uniqueModelNames, Document& doc, Document::AllocatorType& allocator)
		{
			if(!scene)
				return;

			const auto& instances = scene->Instances();
			if(instances.size() == 0)
				return;

			Value jsonInstanceList(kArrayType);
			for(const auto& inst : instances)
			{
				if(!inst->GetModel())
					continue;

				// instance
				Value jsonInst = Value(kObjectType);
				Write(jsonInst, allocator, "name", inst->Name());
				Write(jsonInst, allocator, "model", uniqueModelNames[inst->GetModel()]);
				Value jsonMatrix = Value(kObjectType);
				Write(jsonMatrix, allocator, "matrix", inst->Transform());
				Value jsonTransform = Value(kArrayType);
				jsonTransform.PushBack(jsonMatrix, allocator);
				jsonInst.AddMember("transform", jsonTransform, allocator);

				// add model to array
				jsonInstanceList.PushBack(jsonInst, allocator);
			}

			// add new JSON node to the document
			doc.AddMember("instances", jsonInstanceList, allocator);
		}



		void ExportMaterials(Scene* scene, std::map<std::shared_ptr<Model>, std::string> uniqueModelNames, Document& doc, Document::AllocatorType& allocator)
		{
			if(!scene)
				return;

			const auto& models = scene->Models();
			if(models.size() == 0)
				return;

			Value jsonModelList(kArrayType);
			for(const auto& model : models)
			{
				Value jsonMaterialList = Value(kArrayType);

				for(const auto& mat : model->Materials())
				{
					Value jsonMat = Value(kObjectType);

					// properties
					Write(jsonMat, allocator, "name", mat->Name());
					Write(jsonMat, allocator, "diffuse", mat->Diffuse());
					Write(jsonMat, allocator, "emissive", mat->Emissive());

					// textures
					if (mat->DiffuseMap())
						Write(jsonMat, allocator, "diffuseMap", mat->DiffuseMap()->Path());
					if (mat->NormalMap())
						Write(jsonMat, allocator, "normalMap", mat->NormalMap()->Path());

					// add mat to model material array
					jsonMaterialList.PushBack(jsonMat, allocator);
				}

				// add model to array
				Value jsonModel = Value(kObjectType);
				Write(jsonModel, allocator, "model", uniqueModelNames[model]);
				jsonModel.AddMember("materials", jsonMaterialList, allocator);
				jsonModelList.PushBack(jsonModel, allocator);
			}

			// add new JSON node to the document
			doc.AddMember("materials", jsonModelList, allocator);
		}



		void ExportModels(std::map<std::shared_ptr<Model>, std::string> uniqueModelNames, Document& doc, Document::AllocatorType& allocator)
		{
			Value jsonModelList(kArrayType);
			for(const auto& m : uniqueModelNames)
			{
				Value jsonModel = Value(kObjectType);
				Write(jsonModel, allocator, "name", m.second);
				Write(jsonModel, allocator, "path", m.first->FilePath());

				// add model to array
				jsonModelList.PushBack(jsonModel, allocator);
			}

			// add new JSON node to the document
			doc.AddMember("models", jsonModelList, allocator);
		}



		void ExportRenderSettings(Renderer*  renderer, Document& doc, Document::AllocatorType& allocator)
		{
			if(!renderer)
				return;

			Value jsonRenderer = Value(kObjectType);
			Write(jsonRenderer, allocator, "multisample", renderer->MultiSample());
			Write(jsonRenderer, allocator, "maxdepth", renderer->MaxDepth());
			Write(jsonRenderer, allocator, "aodist", renderer->AODist());
			Write(jsonRenderer, allocator, "zdepthmax", renderer->ZDepthMax());
			Write(jsonRenderer, allocator, "skycolor", renderer->SkyColor());

			// add new JSON node to the document
			doc.AddMember("renderer", jsonRenderer, allocator);
		}



		void ExportPostSettings(Window*  window, Document& doc, Document::AllocatorType& allocator)
		{
			if(!window)
				return;

			const Window::ShaderProperties& postProps = window->PostShaderProperties();

			Value jsonPost = Value(kObjectType);
			Write(jsonPost, allocator, "exposure", postProps.exposure);
			Write(jsonPost, allocator, "gamma", postProps.gamma);

			// add new JSON node to the document
			doc.AddMember("post", jsonPost, allocator);
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// SceneFile
	//--------------------------------------------------------------------------------------------------------------------------
	void SceneFile::Load(const std::string& sceneFile, Scene* scene, CameraNode* camNode, Renderer* renderer, Window* window)
	{
		printf("Loading scene from \"%s\"\n", sceneFile.c_str());

		// read file from disk
		Document doc;
		{
			std::ifstream f(sceneFile);
			IStreamWrapper isw(f);
			doc.ParseStream(isw);
			f.close();
			if(!doc.IsObject())
			{
				printf("\"%s\" does not contain valid json code.", sceneFile.c_str());
				return;
			}
		}

		const auto models = ParseModels(doc);
		const auto instances = ParseInstances(doc);
		ImportModels(scene, models, instances);
		ParseMaterials(scene, doc);
		ParseCamera(camNode, doc);
		ParseRenderSettings(renderer, doc);
		ParsePostSettings(window, doc);
	}



	void SceneFile::Save(const std::string& sceneFile, Scene* scene, CameraNode* camNode, Renderer* renderer, Window* window)
	{
		printf("Saving scene to \"%s\"\n", sceneFile.c_str());

		// create json document
		Document doc;
		doc.SetObject();

		// gather model names
		std::map<std::shared_ptr<Model>, std::string> uniqueModelNames;
		for(auto m : scene->Models())
		{
			std::string name = m->Name();
			int it = 0;
			while(std::find_if(uniqueModelNames.begin(), uniqueModelNames.end(), [&name](auto i) { return name == i.second; }) != uniqueModelNames.end())
			{
				name = format("%s_%i", m->Name().c_str(), ++it);
			}

			uniqueModelNames[m] = name;
		}

		// export to JSON
		Document::AllocatorType& allocator = doc.GetAllocator();
		ExportModels(uniqueModelNames, doc, allocator);
		ExportInstances(scene, uniqueModelNames, doc, allocator);
		ExportMaterials(scene, uniqueModelNames, doc, allocator);
		ExportCamera(camNode, doc, allocator);
		ExportRenderSettings(renderer, doc, allocator);
		ExportPostSettings(window, doc, allocator);

		// write to disk
		std::ofstream f(sceneFile);
		OStreamWrapper osw(f);

		PrettyWriter<OStreamWrapper> writer(osw);
		writer.SetFormatOptions(PrettyFormatOptions::kFormatSingleLineArray);
		doc.Accept(writer);
	}
}