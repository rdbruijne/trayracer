#include "Shader.h"

// project
#include "FileIO/ShaderDescriptor.h"
#include "OpenGL/GLHelpers.h"
#include "OpenGL/GLTexture.h"
#include "Utility/Logger.h"
#include "Utility/Utility.h"

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346 5027)
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// C++
#include <cassert>
#include <memory>

namespace Tracer
{
	namespace
	{
		bool ValidateShader(GLint shaderID, std::string& log)
		{
			GLint result = 0;
			glGetShaderiv(shaderID, GL_COMPILE_STATUS, &result);
			if(result == GL_TRUE)
				return true;

			auto infoLog = std::unique_ptr<char, decltype(free)*>(reinterpret_cast<char*>(malloc(GL_INFO_LOG_LENGTH)), free);
			GLsizei logLength = 0;
			glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &logLength);
			glGetShaderInfoLog(shaderID, logLength, 0, infoLog.get());
			log = infoLog.get();
			Logger::Error("Shader compile error: %s", infoLog.get());
			return false;
		}



		bool ValidateProgram(GLint programID, std::string& log)
		{
			GLint result = 0;
			glGetProgramiv(programID, GL_LINK_STATUS, &result);
			if(result == GL_TRUE)
				return true;

			auto infoLog = std::unique_ptr<char, decltype(free)*>(reinterpret_cast<char*>(malloc(GL_INFO_LOG_LENGTH)), free);
			GLsizei logLength = 0;
			glGetShaderiv(programID, GL_INFO_LOG_LENGTH, &logLength);
			glGetShaderInfoLog(programID, logLength, 0, infoLog.get());
			log = infoLog.get();
			Logger::Error("Shader link error: %s", infoLog.get());
			return false;
		}
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Shader
	//--------------------------------------------------------------------------------------------------------------------------
	Shader::Shader(const std::string& name, const std::string& vertexFile, const std::string& fragmentFile) :
		Named(name),
		mVertexFile(vertexFile),
		mFragmentFile(fragmentFile)
	{
		Compile();
	}



	Shader::~Shader()
	{
		Unload();
	}



	void Shader::Compile()
	{
		// delete old shaders
		Unload();

		// load vertex shader
		std::string vertCode = mVertexFile;
		if(FileExists(mVertexFile))
			vertCode = ReadFile(mVertexFile);
		if(vertCode.empty())
			FatalError("File \"%s\" not found or empty.", mVertexFile.c_str());

		// load fragment shader
		std::string fragCode = mFragmentFile;
		if(FileExists(mFragmentFile))
			fragCode = ReadFile(mFragmentFile);
		if(fragCode.empty())
			FatalError("File \"%s\" not found or empty.", mFragmentFile.c_str());

		// compile vertex shader
		const char* vertCodeCStr = vertCode.c_str();
		GLint vertCodeSize = static_cast<GLint>(vertCode.size());
		mVertexShaderID = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(mVertexShaderID, 1, &vertCodeCStr, &vertCodeSize);
		glCompileShader(mVertexShaderID);
		GL_CHECK();
		if(!ValidateShader(mVertexShaderID, mErrorLog))
		{
			return;
		}

		// compile fragment shader
		const char* fragCodeCStr = fragCode.c_str();
		GLint fragCodeSize = static_cast<GLint>(fragCode.size());
		mFragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(mFragmentShaderID, 1, &fragCodeCStr, &fragCodeSize);
		glCompileShader(mFragmentShaderID);
		GL_CHECK();
		if(!ValidateShader(mFragmentShaderID, mErrorLog))
		{
			return;
		}

		// link shaders
		mShaderID = glCreateProgram();
		glAttachShader(mShaderID, mVertexShaderID);
		glAttachShader(mShaderID, mFragmentShaderID);
		glBindAttribLocation(mShaderID, 0, "pos");
		glLinkProgram(mShaderID);
		GL_CHECK();
		if(!ValidateProgram(mShaderID, mErrorLog))
		{
			return;
		}

		// parse uniforms
		ParseUniforms();

		// load the descriptor
		ParseDescriptor();

		mIsValid = true;
	}



	void Shader::Bind()
	{
		glUseProgram(mShaderID);
	}



	void Shader::Unbind()
	{
		glUseProgram(0);
	}



	void Shader::ApplyUniforms()
	{
		for(auto& [identifier, uniform] : mUniforms)
		{
			switch(uniform.Type())
			{
			case Uniform::Types::Float:
				glUniform1f(glGetUniformLocation(mShaderID, identifier.c_str()), uniform.mData.f);
				break;

			case Uniform::Types::Int:
				glUniform1i(glGetUniformLocation(mShaderID, identifier.c_str()), uniform.mData.i);
				break;

			case Uniform::Types::Texture:
				glActiveTexture(GL_TEXTURE0 + uniform.mData.t.slot);
				if(uniform.mData.t.t)
					uniform.mData.t.t->Bind();
				else
					GLTexture::BindEmpty();
				glUniform1i(glGetUniformLocation(mShaderID, identifier.c_str()), uniform.mData.t.slot);
				break;

			case Uniform::Types::Unknown:
				Logger::Error("Unhandled uniform type: %s (%i)",
							  std::string(magic_enum::enum_name(uniform.Type())).c_str(),
							  static_cast<int>(uniform.Type()));
				break;

			default:
				Logger::Error("Invalid uniform type: %i", static_cast<int>(uniform.Type()));
				break;
			}

			//GL_CHECK();
		}
	}



	Shader::Uniform* Shader::GetUniform(const std::string name)
	{
		std::map<std::string, Uniform>::iterator it = mUniforms.find(name);
		return it == mUniforms.end() ? nullptr : &it->second;
	}



	const Shader::Uniform* Shader::GetUniform(const std::string name) const
	{
		std::map<std::string, Uniform>::const_iterator it = mUniforms.find(name);
		return it == mUniforms.end() ? nullptr : &it->second;
	}



	bool Shader::IsInternalUniform(const std::string& name)
	{
		static const std::array<std::string, 1> sInternals =
		{
			"convergeBuffer"
		};

		return std::find(sInternals.begin(), sInternals.end(), name) != sInternals.end();
	}


	const std::string& Shader::FullScreenQuadFrag()
	{
		static const std::string code =
"#version 430\n\
uniform sampler2D convergeBuffer;\n\
layout(location = 0) in vec2 inUV;\n\
layout(location = 0) out vec4 outColor;\n\
// main shader program\n\
void main()\n\
{\n\
	outColor = vec4(texture(convergeBuffer, inUV).xyz, 1.f);\n\
}";
		return code;
	}



	const std::string& Shader::FullScreenQuadVert()
	{
		static const std::string code =
"#version 430\n\
layout(location = 0) out vec2 outUV;\n\
void main()\n\
{\n\
	// draw single triangle which extends beyond the screen: [-1, -1], [-1, 3], [3, -1]\n\
	const float x = float(gl_VertexID >> 1);\n\
	const float y = float(gl_VertexID & 1);\n\
	outUV = vec2(x * 2.0, y * 2.0);\n\
	gl_Position = vec4(vec2(x * 4.0 - 1.0, y * 4.0 - 1.0), 0, 1.0);\n\
}";
		return code;
	}



	void Shader::Unload()
	{
		if(mShaderID)
		{
			glDetachShader(mShaderID, mVertexShaderID);
			glDetachShader(mShaderID, mFragmentShaderID);
			glDeleteProgram(mShaderID);
		}

		if(mVertexShaderID)
			glDeleteShader(mVertexShaderID);

		if(mFragmentShaderID)
			glDeleteShader(mFragmentShaderID);

		mVertexShaderID = 0;
		mFragmentShaderID = 0;
		mShaderID = 0;
	}



	void Shader::ParseUniforms()
	{
		// add uniforms
		auto AddUniform = [this](const std::string& name, Uniform::Types type) -> Uniform&
		{
			std::map<std::string, Uniform>::iterator it = mUniforms.find(name);
			if(it != mUniforms.end())
				return mUniforms.find(name)->second;

			return mUniforms.emplace(name, type).first->second;
		};



		GLint count;
		glGetProgramiv(mShaderID, GL_ACTIVE_UNIFORMS, &count);

		GLint size;
		GLenum type;
		constexpr GLsizei bufSize = 1024;
		GLchar name[bufSize];
		GLsizei length;

		for(GLint ix = 0; ix < count; ix++)
		{
			glGetActiveUniform(mShaderID, (GLuint)ix, bufSize, &length, &size, &type, name);
			switch(type)
			{
			case GL_FLOAT:
				{
					float f;
					glGetUniformfv(mShaderID, glGetUniformLocation(mShaderID, name), &f);
					Uniform& u = AddUniform(name, Uniform::Types::Float);
					u.Set(f);
				}
				break;

			case GL_INT:
				{
					int i;
					glGetUniformiv(mShaderID, glGetUniformLocation(mShaderID, name), &i);
					Uniform& u = AddUniform(name, Uniform::Types::Int);
					u.Set(i);
				}
				break;

			case GL_SAMPLER_2D:
				AddUniform(name, Uniform::Types::Texture);
				break;
			}

			GL_CHECK();
		}
	}



	void Shader::ParseDescriptor()
	{
		// only execute if mFragmentFile is a file path
		if(!FileExists(mFragmentFile))
			return;

		const std::string descriptorPath = ReplaceExtension(mFragmentFile, ".json");
		if(FileExists(descriptorPath))
			ShaderDescriptor::Load(descriptorPath, this);
		else
			ShaderDescriptor::Save(descriptorPath, this);
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Shader Uniform
	//--------------------------------------------------------------------------------------------------------------------------
	// float
	void Shader::Uniform::Get(float* v) const
	{
		assert(v);
		assert(mType == Types::Float);
		*v = mData.f;
	}



	void Shader::Uniform::Set(float v)
	{
		assert(mType == Types::Float);
		mData.f = v;
	}



	void Shader::Uniform::GetRange(float* min, float* max) const
	{
		assert(min);
		assert(max);
		assert(mType == Types::Float);
		assert(mHasRange);
		*min = mRange.f[0];
		*max = mRange.f[1];
	}



	void Shader::Uniform::SetRange(float min, float max)
	{
		assert(mType == Types::Float);
		mRange.f[0] = min;
		mRange.f[1] = max;
		mHasRange = true;
	}



	// int
	void Shader::Uniform::Get(int* v) const
	{
		assert(v);
		assert(mType == Types::Int);
		*v = mData.i;
	}



	void Shader::Uniform::Set(int v)
	{
		assert(mType == Types::Int);
		mData.i = v;
	}



	void Shader::Uniform::GetRange(int* min, int* max) const
	{
		assert(min);
		assert(max);
		assert(mType == Types::Int);
		assert(mHasRange);
		*min = mRange.i[0];
		*max = mRange.i[1];
	}



	void Shader::Uniform::SetRange(int min, int max)
	{
		assert(mType == Types::Int);
		mRange.i[0] = min;
		mRange.i[1] = max;
		mHasRange = true;
	}



	// texture
	void Shader::Uniform::Get(int* slot, GLTexture** tex) const
	{
		assert(slot);
		assert(tex);
		assert(mType == Types::Texture);
		*slot = mData.t.slot;
		*tex = mData.t.t;
	}



	void Shader::Uniform::Set(int slot, GLTexture* tex)
	{
		assert(mType == Types::Texture);
		mData.t.slot = slot;
		mData.t.t = tex;
	}
}
