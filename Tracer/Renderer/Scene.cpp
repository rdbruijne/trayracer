#include "Scene.h"

// Project
#include "Resources/CameraNode.h"
#include "Resources/Model.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Utility/Importer.h"

// RapidJson
#pragma warning(push)
#pragma warning(disable: 4061 4464)
#include "RapidJson/document.h"
#pragma warning(pop)

// C++
#include <fstream>
#include <map>
#include <set>
#include <streambuf>
#include <string>

namespace Tracer
{

	void Scene::Load(const std::string& sceneFile, CameraNode& camNode)
	{
		// read file from disk
		std::ifstream f(sceneFile);
		const std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		f.close();

		if(json.empty())
		{
			printf("\"%s\" does not contain json code.", sceneFile.c_str());
			return;
		}

		// parse the json
		rapidjson::Document doc;
		doc.Parse(json.c_str(), json.length());
		if(!doc.IsObject())
		{
			printf("\"%s\" does not contain valid json code.", sceneFile.c_str());
			return;
		}

		// models/instances
		if(doc.HasMember("models"))
		{
			const rapidjson::Value& jsonModels = doc["models"];
			if(jsonModels.IsArray())
			{
				std::map<std::string, std::shared_ptr<Model>> importedModels;

				for(rapidjson::SizeType modelIx = 0; modelIx < jsonModels.Size(); modelIx++)
				{
					const rapidjson::Value& jsonInstance = jsonModels[modelIx];

					// check for required fields
					if(!jsonInstance.HasMember("model"))
						continue;

					// model path
					const std::string modelPath = jsonInstance["model"].GetString();

					// transform
					float3x4 modelTransform = make_float3x4(); // identity
					if(jsonInstance.HasMember("transform"))
					{
						const rapidjson::Value& jsonTransformOperations = jsonInstance["transform"];
						for(rapidjson::SizeType transOpIx = 0; transOpIx < jsonTransformOperations.Size(); transOpIx++)
						{
							const rapidjson::Value& op = jsonTransformOperations[transOpIx];
							if(op.HasMember("matrix"))
							{
								const rapidjson::Value& opVal = op["matrix"];
								if(opVal.IsArray() && opVal.Size() == 12)
								{
									float transValues[12] = {};
									for(rapidjson::SizeType i = 0; i < opVal.Size(); i++)
										transValues[i] = opVal[i].GetFloat();
									const float3x4 t = make_float3x4(
										transValues[0], transValues[1], transValues[ 2], transValues[ 3],
										transValues[4], transValues[5], transValues[ 6], transValues[ 7],
										transValues[8], transValues[9], transValues[10], transValues[11]);
									modelTransform *= t;
								}
							}
							else if(op.HasMember("translate"))
							{
								const rapidjson::Value& opVal = op["translate"];
								if(opVal.IsArray() && opVal.Size() == 3)
									modelTransform *= translate_3x4(opVal[0].GetFloat(), opVal[1].GetFloat(), opVal[2].GetFloat());
							}
							else if(op.HasMember("scale"))
							{
								const rapidjson::Value& opVal = op["scale"];
								if(opVal.IsArray() && opVal.Size() == 3)
									modelTransform *= scale_3x4(opVal[0].GetFloat(), opVal[1].GetFloat(), opVal[2].GetFloat());

							}
							else if(op.HasMember("rotate-x"))
							{
								const rapidjson::Value& opVal = op["rotate-x"];
								if(opVal.IsFloat())
									modelTransform *= rotate_x_3x4(opVal.GetFloat());
							}
							else if(op.HasMember("rotate-y"))
							{
								const rapidjson::Value& opVal = op["rotate-y"];
								if(opVal.IsFloat())
									modelTransform *= rotate_y_3x4(opVal.GetFloat());
							}
							else if(op.HasMember("rotate-z"))
							{
								const rapidjson::Value& opVal = op["rotate-z"];
								if(opVal.IsFloat())
									modelTransform *= rotate_z_3x4(opVal.GetFloat());
							}
						}
					}

					// instance name
					std::string instName = FileNameWithoutExtension(modelPath);
					if(jsonInstance.HasMember("name"))
						instName = jsonInstance["name"].GetString();

					// find model for given path
					std::shared_ptr<Model> model = nullptr;
					auto it = std::find_if(mModels.begin(), mModels.end(), [modelPath](std::shared_ptr<Model> m) { return m->FilePath() == modelPath; });
					if(it != mModels.end())
					{
						model = *it;
					}
					else
					{
						// import the model
						model = ImportModel(modelPath);
						AddModel(model);
					}

					// add instance
					AddInstance(std::make_shared<Instance>(instName, model, modelTransform));
				}
			}
		}

		// camera
		if(doc.HasMember("camera"))
		{
			float3 pos = make_float3(0, 0, -1);
			float3 target = make_float3(0, 0, 0);
			float3 up = make_float3(0, 1, 0);
			float fov = 90.f;

			const rapidjson::Value& jsonCamera = doc["camera"];

			// position
			if(jsonCamera.HasMember("position"))
			{
				const rapidjson::Value& jsonPos = jsonCamera["position"];
				if(jsonPos.IsArray() && jsonPos.Size() == 3)
				{
					pos.x = jsonPos[0].GetFloat();
					pos.y = jsonPos[1].GetFloat();
					pos.z = jsonPos[2].GetFloat();
				}
			}

			// target
			if(jsonCamera.HasMember("target"))
			{
				const rapidjson::Value& jsonTarget = jsonCamera["target"];
				if(jsonTarget.IsArray() && jsonTarget.Size() == 3)
				{
					target.x = jsonTarget[0].GetFloat();
					target.y = jsonTarget[1].GetFloat();
					target.z = jsonTarget[2].GetFloat();
				}
			}

			// up
			if(jsonCamera.HasMember("position"))
			{
				const rapidjson::Value& jsonUp = jsonCamera["up"];
				if(jsonUp.IsArray() && jsonUp.Size() == 3)
				{
					up.x = jsonUp[0].GetFloat();
					up.y = jsonUp[1].GetFloat();
					up.z = jsonUp[2].GetFloat();
				}
			}

			// fov
			if(jsonCamera.HasMember("fov"))
			{
				fov = jsonCamera["fov"].GetFloat();
				if(fov < .1f || fov >= 179.9f)
				{
					printf("FOV must be between 0 and 180, correcting.");
					fov = fminf(fmaxf(fov, .1f), 179.9f);
				}
			}

			if(pos == target)
			{
				printf("Camera position & target cannot overlap, correcting.");
				target = pos + make_float3(0, 0, 1);
			}

			// set camera
			camNode = CameraNode(pos, target, up, fov * DegToRad);
		}
	}



	bool Scene::IsDirty() const
	{
		if(mIsDirty)
			return true;

		for(auto& i : mInstances)
			if(i->IsDirty())
				return true;

		return false;
	}



	size_t Scene::InstanceCount() const
	{
		return mInstances.size();
	}



	size_t Scene::MaterialCount() const
	{
		size_t matCount = 0;
		for(auto m : mModels)
			matCount += m->Materials().size();
		return matCount;
	}



	size_t Scene::ModelCount() const
	{
		return mModels.size();
	}



	size_t Scene::InstancedModelCount() const
	{
		std::set<std::shared_ptr<Model>> models;
		for(auto& i : mInstances)
			models.insert(i->GetModel());
		return models.size();
	}



	size_t Scene::TriangleCount() const
	{
		size_t triCount = 0;
		for(auto& i : mInstances)
			triCount += i->GetModel()->PolyCount();
		return triCount;
	}



	size_t Scene::UniqueTriangleCount() const
	{
		std::set<std::shared_ptr<Model>> models;
		for(auto& i : mInstances)
			models.insert(i->GetModel());

		size_t triCount = 0;
		for(const auto& m : models)
			triCount += m->PolyCount();
		return triCount;
	}



	void Scene::AddModel(std::shared_ptr<Model> model)
	{
		if(model && std::find(mModels.begin(), mModels.end(), model) == mModels.end())
		{
			mModels.push_back(model);
			MarkDirty();
		}
	}



	void Scene::AddInstance(std::shared_ptr<Instance> instance)
	{
		assert(instance->GetModel() != nullptr);
		assert(std::find(mModels.begin(), mModels.end(), instance->GetModel()) != mModels.end());

		if(instance && std::find(mInstances.begin(), mInstances.end(), instance) == mInstances.end())
		{
			mInstances.push_back(instance);
			MarkDirty();
		}
	}



	std::shared_ptr<Tracer::Material> Scene::GetMaterial(uint32_t instanceIx, uint32_t primIx)
	{
		if(instanceIx >= mInstances.size())
			return nullptr;
		return mInstances[instanceIx]->GetModel()->GetMaterial(primIx);
	}
}
