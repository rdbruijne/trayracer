#include "SceneFile.h"

// Project
#include "FileIO/Importer.h"
#include "OpenGL/Window.h"
#include "Renderer/Renderer.h"
#include "Renderer/Scene.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
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
#include <streambuf>

// #TODO: clean code by adding helper functions in nameless namespace

namespace Tracer
{
	void SceneFile::Load(const std::string& sceneFile, Scene* scene, CameraNode* camNode, Renderer* renderer, Window* window)
	{
		using namespace rapidjson;

		printf("Loading scene from \"%s\"\n", sceneFile.c_str());

		// read file from disk
		std::ifstream f(sceneFile);
		IStreamWrapper isw(f);
		Document doc;
		doc.ParseStream(isw);
		f.close();
		if(!doc.IsObject())
		{
			printf("\"%s\" does not contain valid json code.", sceneFile.c_str());
			return;
		}

		// models/instances
		if(scene)
		{
			if(doc.HasMember("models"))
			{
				const Value& jsonModels = doc["models"];
				if(jsonModels.IsArray())
				{
					std::map<std::string, std::shared_ptr<Model>> importedModels;

					for(SizeType modelIx = 0; modelIx < jsonModels.Size(); modelIx++)
					{
						const Value& jsonInstance = jsonModels[modelIx];

						// check for required fields
						if(!jsonInstance.HasMember("model"))
							continue;

						// model path
						const std::string modelPath = jsonInstance["model"].GetString();

						// transform
						float3x4 modelTransform = make_float3x4(); // identity
						if(jsonInstance.HasMember("transform"))
						{
							const Value& jsonTransformOperations = jsonInstance["transform"];
							for(rapidjson::SizeType transOpIx = 0; transOpIx < jsonTransformOperations.Size(); transOpIx++)
							{
								const Value& op = jsonTransformOperations[transOpIx];
								if(op.HasMember("matrix"))
								{
									const Value& opVal = op["matrix"];
									if(opVal.IsArray() && opVal.Size() == 12)
									{
										float transValues[12] = {};
										for(SizeType i = 0; i < opVal.Size(); i++)
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
									const Value& opVal = op["translate"];
									if(opVal.IsArray() && opVal.Size() == 3)
										modelTransform *= translate_3x4(opVal[0].GetFloat(), opVal[1].GetFloat(), opVal[2].GetFloat());
								}
								else if(op.HasMember("scale"))
								{
									const Value& opVal = op["scale"];
									if(opVal.IsArray() && opVal.Size() == 3)
										modelTransform *= scale_3x4(opVal[0].GetFloat(), opVal[1].GetFloat(), opVal[2].GetFloat());
									else if(opVal.IsNumber())
										modelTransform *= scale_3x4(opVal.GetFloat());
								}
								else if(op.HasMember("rotate-x"))
								{
									const Value& opVal = op["rotate-x"];
									modelTransform *= rotate_x_3x4(opVal.GetFloat() * DegToRad);
								}
								else if(op.HasMember("rotate-y"))
								{
									const Value& opVal = op["rotate-y"];
									modelTransform *= rotate_y_3x4(opVal.GetFloat() * DegToRad);
								}
								else if(op.HasMember("rotate-z"))
								{
									const Value& opVal = op["rotate-z"];
									modelTransform *= rotate_z_3x4(opVal.GetFloat() * DegToRad);
								}
							}
						}

						// instance name
						std::string instName = FileNameWithoutExtension(modelPath);
						if(jsonInstance.HasMember("name"))
							instName = jsonInstance["name"].GetString();

						// find model for given path
						std::shared_ptr<Model> model = nullptr;
						auto it = importedModels.find(modelPath);
						if(it != importedModels.end())
						{
							model = it->second;
						}
						else
						{
							// import the model
							model = Importer::ImportModel(modelPath);
							scene->AddModel(model);
						}

						// add instance
						scene->AddInstance(std::make_shared<Instance>(instName, model, modelTransform));
					}
				}
			}
		}

		// camera
		if(camNode && doc.HasMember("camera"))
		{
			float3 pos = make_float3(0, 0, -1);
			float3 target = make_float3(0, 0, 0);
			float3 up = make_float3(0, 1, 0);
			float fov = 90.f;

			const Value& jsonCamera = doc["camera"];

			// position
			if(jsonCamera.HasMember("position"))
			{
				const Value& jsonPos = jsonCamera["position"];
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
				const Value& jsonTarget = jsonCamera["target"];
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
				const Value& jsonUp = jsonCamera["up"];
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
			*camNode = CameraNode(pos, target, up, fov * DegToRad);
		}

		// render settings
		if(renderer && doc.HasMember("renderer"))
		{
			const Value& jsonRenderer = doc["renderer"];

			// multi-sample
			if(jsonRenderer.HasMember("multisample"))
				renderer->SetMultiSample(jsonRenderer["multisample"].GetInt());

			// max depth
			if(jsonRenderer.HasMember("maxdepth"))
				renderer->SetMaxDepth(jsonRenderer["maxdepth"].GetInt());

			// ao dist
			if(jsonRenderer.HasMember("aodist"))
				renderer->SetAODist(jsonRenderer["aodist"].GetFloat());

			// z-depth max
			if(jsonRenderer.HasMember("zdepthmax"))
				renderer->SetZDepthMax(jsonRenderer["zdepthmax"].GetFloat());

			// sky color
			const Value& jsonSkyColor = jsonRenderer["skycolor"];
			if(jsonSkyColor.IsArray() && jsonSkyColor.Size() == 3)
				renderer->SetSkyColor(make_float3(jsonSkyColor[0].GetFloat(), jsonSkyColor[1].GetFloat(), jsonSkyColor[2].GetFloat()));
		}

		// post
		if(window && doc.HasMember("post"))
		{
			Window::ShaderProperties postProps = window->PostShaderProperties();
			const Value& jsonPost = doc["post"];

			// exposure
			if(jsonPost.HasMember("exposure"))
				postProps.exposure = jsonPost["exposure"].GetFloat();

			// gamma
			if(jsonPost.HasMember("gamma"))
				postProps.gamma = jsonPost["gamma"].GetFloat();

			window->SetPostShaderProperties(postProps);
		}
	}



	void SceneFile::Save(const std::string& sceneFile, Scene* scene, CameraNode* camNode, Renderer* renderer, Window* window)
	{
		using namespace rapidjson;

		printf("Saving scene to \"%s\"\n", sceneFile.c_str());

		// create json document
		Document doc;
		doc.SetObject();

		Document::AllocatorType& allocator = doc.GetAllocator();

		// models/instances
		if(scene)
		{
			auto& instances = scene->Instances();
			if(instances.size() > 0)
			{
				Value jsonModels(kArrayType);
				for(auto inst : instances)
				{
					if(!inst->GetModel())
						continue;

					// instance
					Value jsonInst = Value(kObjectType);

					// name
					Value jsonName = Value(kObjectType);
					const std::string instName = inst->Name();
					jsonName.SetString(instName.c_str(), static_cast<SizeType>(instName.length()), allocator);
					jsonInst.AddMember("name", jsonName, allocator);

					// file path
					Value jsonPath = Value(kObjectType);
					const std::string filePath = inst->GetModel()->FilePath();
					jsonPath.SetString(filePath.c_str(), static_cast<SizeType>(filePath.length()), allocator);
					jsonInst.AddMember("model", jsonPath, allocator);

					// transform
					Value jsonMatrix = Value(kArrayType);
					const float3x4& t = inst->Transform();
					float tempVals[12] = { t.x.x, t.x.y, t.x.z, t.y.x, t.y.y, t.y.z, t.z.x, t.z.y, t.z.z, t.tx, t.ty, t.tz };
					for(int i = 0; i < 12; i++)
					{
						Value f = Value(kObjectType);
						f.SetFloat(tempVals[i]);
						jsonMatrix.PushBack(f, allocator);
					}

					Value jsonMatrixObj = Value(kObjectType);
					jsonMatrixObj.AddMember("matrix", jsonMatrix, allocator);

					Value jsonTransform = Value(kArrayType);
					jsonTransform.PushBack(jsonMatrixObj, allocator);

					jsonInst.AddMember("transform", jsonTransform, allocator);

					// add model to array
					jsonModels.PushBack(jsonInst, allocator);
				}
				doc.AddMember("models", jsonModels, allocator);
			}
		}

		// camera
		if(camNode)
		{
			Value x = Value(kObjectType);
			Value y = Value(kObjectType);
			Value z = Value(kObjectType);

			// position
			Value jsonPos = Value(kArrayType);
			const float3& camPos = camNode->Position();
			x.SetFloat(camPos.x);
			y.SetFloat(camPos.y);
			z.SetFloat(camPos.z);
			jsonPos.PushBack(x, allocator);
			jsonPos.PushBack(y, allocator);
			jsonPos.PushBack(z, allocator);

			// target
			Value jsonTarget = Value(kArrayType);
			const float3& camTarget = camNode->Target();
			x.SetFloat(camTarget.x);
			y.SetFloat(camTarget.y);
			z.SetFloat(camTarget.z);
			jsonTarget.PushBack(x, allocator);
			jsonTarget.PushBack(y, allocator);
			jsonTarget.PushBack(z, allocator);

			// up
			Value jsonUp = Value(kArrayType);
			const float3& camUp = camNode->Up();
			x.SetFloat(camUp.x);
			y.SetFloat(camUp.y);
			z.SetFloat(camUp.z);
			jsonUp.PushBack(x, allocator);
			jsonUp.PushBack(y, allocator);
			jsonUp.PushBack(z, allocator);

			// fov
			Value jsonFov = Value(kObjectType);
			jsonFov.SetFloat(camNode->Fov() * RadToDeg);

			// create json camera
			Value jsonCam = Value(kObjectType);
			jsonCam.AddMember("position", jsonPos, allocator);
			jsonCam.AddMember("target", jsonTarget, allocator);
			jsonCam.AddMember("up", jsonUp, allocator);
			jsonCam.AddMember("fov", jsonFov, allocator);

			// add to doc
			doc.AddMember("camera", jsonCam, allocator);
		}

		// render settings
		if(renderer)
		{
			Value jsonRenderSettings = Value(kObjectType);

			// multi-sample
			Value jsonMultiSample = Value(kObjectType);
			jsonMultiSample.SetInt(renderer->MultiSample());
			jsonRenderSettings.AddMember("multisample", jsonMultiSample, allocator);

			// max depth
			Value jsonMaxDepth = Value(kObjectType);
			jsonMaxDepth.SetInt(renderer->MaxDepth());
			jsonRenderSettings.AddMember("maxdepth", jsonMaxDepth, allocator);

			// ao dist
			Value jsonAoDist = Value(kObjectType);
			jsonAoDist.SetFloat(renderer->AODist());
			jsonRenderSettings.AddMember("aodist", jsonAoDist, allocator);

			// z-depth max
			Value jsonZDepthMax = Value(kObjectType);
			jsonZDepthMax.SetFloat(renderer->ZDepthMax());
			jsonRenderSettings.AddMember("zdepthmax", jsonZDepthMax, allocator);

			// sky color
			const float3& skyColor = renderer->SkyColor();

			Value jsonSkyColorR = Value(kObjectType);
			jsonSkyColorR.SetFloat(skyColor.x);

			Value jsonSkyColorG = Value(kObjectType);
			jsonSkyColorG.SetFloat(skyColor.y);

			Value jsonSkyColorB = Value(kObjectType);
			jsonSkyColorB.SetFloat(skyColor.z);

			Value jsonSkyColor = Value(kArrayType);
			jsonSkyColor.PushBack(jsonSkyColorR, allocator);
			jsonSkyColor.PushBack(jsonSkyColorG, allocator);
			jsonSkyColor.PushBack(jsonSkyColorB, allocator);
			jsonRenderSettings.AddMember("skycolor", jsonSkyColor, allocator);

			// add to doc
			doc.AddMember("renderer", jsonRenderSettings, allocator);
		}

		// post settings
		if(window)
		{
			const Window::ShaderProperties& postProps = window->PostShaderProperties();
			Value jsonPostSettings = Value(kObjectType);

			// exposure
			Value jsonExposure = Value(kObjectType);
			jsonExposure.SetFloat(postProps.exposure);
			jsonPostSettings.AddMember("exposure", jsonExposure, allocator);

			// gamma
			Value jsonGamma = Value(kObjectType);
			jsonGamma.SetFloat(postProps.gamma);
			jsonPostSettings.AddMember("gamma", jsonGamma, allocator);

			// add to doc
			doc.AddMember("post", jsonPostSettings, allocator);
		}

		// write to disk
		std::ofstream f(sceneFile);
		OStreamWrapper osw(f);

		PrettyWriter<OStreamWrapper> writer(osw);
		doc.Accept(writer);
	}
}
