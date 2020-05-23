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

			Value jsonVectorObj = Value(kObjectType);
			jsonVectorObj.AddMember(StringRef(memberName), jsonVector, allocator);
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

			Value jsonMatrixObj = Value(kObjectType);
			jsonValue.AddMember(StringRef(memberName), jsonMatrix, allocator);
		}



		void Write(Value& jsonValue, Document::AllocatorType& allocator, const char* memberName, const std::string& val)
		{
			Value v = Value(kObjectType);
			v.SetString(val.c_str(), static_cast<SizeType>(val.length()), allocator);
			jsonValue.AddMember(StringRef(memberName), v, allocator);
		}
	}



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

		// reusable data for json reading
		int i;
		float f;
		float3 f3;
		float3x4 f3x4;

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

						// model path
						std::string modelPath;
						if(!Read(jsonInstance, "model", modelPath))
							continue;

						// transform
						float3x4 modelTransform = make_float3x4(); // identity
						if(jsonInstance.HasMember("transform"))
						{
							const Value& jsonTransformOperations = jsonInstance["transform"];
							for(SizeType transOpIx = 0; transOpIx < jsonTransformOperations.Size(); transOpIx++)
							{
								const Value& op = jsonTransformOperations[transOpIx];
								if(Read(op, "matrix", f3x4))
									modelTransform *= f3x4;
								else if(Read(op, "translate", f3))
									modelTransform *= translate_3x4(f3);
								else if(Read(op, "rotate-x", f))
									modelTransform *= rotate_x_3x4(f * DegToRad);
								else if(Read(op, "rotate-y", f))
									modelTransform *= rotate_y_3x4(f * DegToRad);
								else if(Read(op, "rotate-z", f))
									modelTransform *= rotate_z_3x4(f * DegToRad);
								else if(Read(op, "scale", f))
									modelTransform *= scale_3x4(f);
								else if(Read(op, "scale", f3))
									modelTransform *= scale_3x4(f3);
							}
						}

						// instance name
						std::string instName = FileNameWithoutExtension(modelPath);
						Read(jsonInstance, "name", instName);

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

		// render settings
		if(renderer && doc.HasMember("renderer"))
		{
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

		// post
		if(window && doc.HasMember("post"))
		{
			Window::ShaderProperties postProps = window->PostShaderProperties();
			const Value& jsonPost = doc["post"];

			Read(jsonPost, "exposure", postProps.exposure);
			Read(jsonPost, "gamma", postProps.gamma);

			window->SetPostShaderProperties(postProps);
		}
	}



	void SceneFile::Save(const std::string& sceneFile, Scene* scene, CameraNode* camNode, Renderer* renderer, Window* window)
	{
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
					Write(jsonInst, allocator, "name", inst->Name());
					Write(jsonInst, allocator, "model", inst->GetModel()->FilePath());

					Value jsonTransform = Value(kArrayType);
					Write(jsonTransform, allocator, "matrix", inst->Transform());

					// add model to array
					jsonModels.PushBack(jsonInst, allocator);
				}
				doc.AddMember("models", jsonModels, allocator);
			}
		}

		// camera
		if(camNode)
		{
			Value jsonCam = Value(kObjectType);
			Write(jsonCam, allocator, "position", camNode->Position());
			Write(jsonCam, allocator, "target", camNode->Target());
			Write(jsonCam, allocator, "up", camNode->Up());
			Write(jsonCam, allocator, "fov", camNode->Fov());
			doc.AddMember("camera", jsonCam, allocator);
		}

		// render settings
		if(renderer)
		{
			Value jsonRenderer = Value(kObjectType);
			Write(jsonRenderer, allocator, "multisample", renderer->MultiSample());
			Write(jsonRenderer, allocator, "maxdepth", renderer->MaxDepth());
			Write(jsonRenderer, allocator, "aodist", renderer->AODist());
			Write(jsonRenderer, allocator, "zdepthmax", renderer->ZDepthMax());
			Write(jsonRenderer, allocator, "skycolor", renderer->SkyColor());
			doc.AddMember("renderer", jsonRenderer, allocator);
		}

		// post settings
		if(window)
		{
			const Window::ShaderProperties& postProps = window->PostShaderProperties();

			Value jsonPost = Value(kObjectType);
			Write(jsonPost, allocator, "exposure", postProps.exposure);
			Write(jsonPost, allocator, "gamma", postProps.gamma);
			doc.AddMember("post", jsonPost, allocator);
		}

		// write to disk
		std::ofstream f(sceneFile);
		OStreamWrapper osw(f);

		PrettyWriter<OStreamWrapper> writer(osw);
		doc.Accept(writer);
	}
}
