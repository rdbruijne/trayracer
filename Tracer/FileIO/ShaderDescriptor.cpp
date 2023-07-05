#include "ShaderDescriptor.h"

// Project
#include "FileIO/JsonHelpers.h"
#include "Logging/Logger.h"
#include "OpenGL/Shader.h"
#include "Utility/LinearMath.h"
#include "Utility/FileSystem.h"
#include "Utility/Strings.h"
#include "Utility/Stopwatch.h"

// C++
#include <fstream>

using namespace rapidjson;

// Keys
#define Key_Enum			"enum"
#define Key_Name			"name"
#define Key_Shader			"shader"
#define Key_Uniforms		"uniforms"
#define Key_Path			"path"
#define Key_Range			"range"
#define Key_Value			"value"
#define Key_Type			"type"
#define Key_Logarithmic		"logarithmic"

// #TODO: import/export texture path

namespace Tracer
{
	namespace
	{
		void ExportShader(Shader* shader, Document& doc, Document::AllocatorType& allocator)
		{
			if(!shader)
				return;

			// export basic info
			doc.AddMember(Key_Name, Value(shader->Name().data(), allocator), allocator);

			// export uniforms
			Value jsonUniformArray = Value(kArrayType);
			const std::map<std::string, Shader::Uniform>& shaderUniforms = shader->Uniforms();
			for(const auto [name, uniform] : shaderUniforms)
			{
				if(Shader::IsInternalUniform(name))
					continue;

				Value jsonUniform = Value(kObjectType);
				Write(jsonUniform, allocator, Key_Name, name);
				Write(jsonUniform, allocator, Key_Type, uniform.Type());
				if(uniform.HasRange())
				{
					if(uniform.Type() == Shader::Uniform::Types::Float)
					{
						float2 range;
						uniform.GetRange(&range.x, &range.y);
						Write(jsonUniform, allocator, Key_Range, range);
					}
					else if(uniform.Type() == Shader::Uniform::Types::Int)
					{
						int2 range;
						uniform.GetRange(&range.x, &range.y);
						Write(jsonUniform, allocator, Key_Range, range);

						const std::vector<std::string>& enumKeys = uniform.EnumKeys();
						if(enumKeys.size() != 0)
						{
							Value jsonEnumKeys = Value(kArrayType);
							for(const std::string& k : enumKeys)
								jsonEnumKeys.PushBack(Value(k.data(), allocator), allocator);
							Value key = Value(Key_Enum, allocator);
							jsonUniform.AddMember(key, jsonEnumKeys, allocator);
						}
					}

					// logarithmic slider
					Write(jsonUniform, allocator, Key_Logarithmic, uniform.IsLogarithmic());
				}

				// add uniform to array
				jsonUniformArray.PushBack(jsonUniform, allocator);
			}
			doc.AddMember(Key_Uniforms, jsonUniformArray, allocator);
		}



		void ParseShader(Shader* shader, Document& doc)
		{
			if(!shader)
				return;

			// basic info
			if(doc.HasMember(Key_Name))
			{
				const Value& shaderName = doc[Key_Name];
				shader->SetName(shaderName.GetString());
			}

			// uniforms
			const Value& jsonUniformArray = doc[Key_Uniforms];
			if(!jsonUniformArray.IsArray())
				return;

			for(SizeType uniformIx = 0; uniformIx < jsonUniformArray.Size(); uniformIx++)
			{
				const Value& jsonUniform = jsonUniformArray[uniformIx];

				// fetch the uniform by name
				std::string uniformName;
				if(!Read(jsonUniform, Key_Name, uniformName))
					continue;

				if(Shader::IsInternalUniform(uniformName))
					continue;

				if(shader->Uniforms().count(uniformName) == 0)
					continue;

				Shader::Uniform* uniform = shader->GetUniform(uniformName);
				if(!uniform)
					continue;

				// check the type
				Shader::Uniform::Types uniformType;
				if(!Read(jsonUniform, Key_Type, uniformType))
					continue;

				if(uniformType != uniform->Type())
					continue;

				// get a range
				if(jsonUniform.HasMember(Key_Range))
				{
					if(uniformType == Shader::Uniform::Types::Float)
					{
						float2 range;
						Read(jsonUniform, Key_Range, range);
						uniform->SetRange(range.x, range.y);
					}
					else if(uniformType == Shader::Uniform::Types::Int)
					{
						int2 range;
						Read(jsonUniform, Key_Range, range);
						uniform->SetRange(range.x, range.y);
					}

					// logarithmic slider
					bool logarithmic;
					if(Read(jsonUniform, Key_Logarithmic, logarithmic))
						uniform->SetLogarithmic(logarithmic);
				}

				// enum keys
				if(jsonUniform.HasMember(Key_Enum))
				{
					const Value& val = jsonUniform[Key_Enum];
					if(val.IsArray())
					{
						std::vector<std::string> keys;
						keys.reserve(val.Size());
						for(SizeType i = 0; i < val.Size(); i++)
						{
							if(!val[i].IsString())
								continue;
							keys.push_back(val[i].GetString());
						}
						uniform->SetEnumKeys(keys);
					}
				}
			}
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// ShaderDescriptor
	//--------------------------------------------------------------------------------------------------------------------------
	bool ShaderDescriptor::Load(const std::string& descriptorFile, Shader* shader)
	{
		Logger::Info("Loading shader descriptor from \"%s\"", descriptorFile.c_str());
		Stopwatch sw;

		// read file from disk
		Document doc;
		{
			std::ifstream f(descriptorFile);
			if(!f.is_open())
			{
				Logger::Error("Failed to open \"%s\" for reading.", descriptorFile.c_str());
				return false;
			}

			IStreamWrapper isw(f);
			doc.ParseStream(isw);
			f.close();
			if(!doc.IsObject())
			{
				Logger::Error("\"%s\" does not contain valid json code.", descriptorFile.c_str());
				return false;
			}
		}

		if(shader)
		{
			ParseShader(shader, doc);
		}

		Logger::Info("Loaded shader descriptor in %s", sw.ElapsedString().c_str());
		return true;
	}



	bool ShaderDescriptor::Save(const std::string& descriptorFile, Shader* shader)
	{
		std::string globalPath = GlobalPath(descriptorFile);
		if(ToLower(FileExtension(globalPath)) != ".json")
			globalPath += ".json";

		Logger::Info("Saving shader descriptor to \"%s\"", globalPath.c_str());
		Stopwatch sw;

		// create json document
		Document doc;
		Document::AllocatorType& allocator = doc.GetAllocator();
		doc.SetObject();

		if(shader)
		{
			ExportShader(shader, doc, allocator);
		}

		// write to disk
		std::ofstream f(globalPath);
		if(!f.is_open())
		{
			Logger::Error("Failed to open \"%s\" for writing.", descriptorFile.c_str());
			return false;
		}

		OStreamWrapper osw(f);
		PrettyWriter<OStreamWrapper> writer(osw);
		writer.SetFormatOptions(PrettyFormatOptions::kFormatSingleLineArray);
		if(!doc.Accept(writer))
		{
			Logger::Error("Failed to write shader descriptor to \"%s\".", descriptorFile.c_str());
			return false;
		}

		Logger::Info("Saved shader descriptor in %s", sw.ElapsedString().c_str());
		return true;
	}
}
