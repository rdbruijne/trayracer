#include "Shader.h"

// project
#include "OpenGL/GLHelpers.h"
#include "OpenGL/GLTexture.h"
#include "Utility/Utility.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// C++
#include <stdexcept>

namespace Tracer
{
	namespace
	{
		void CheckShader(GLint shaderID)
		{
			GLint result = 0;
			glGetShaderiv(shaderID, GL_COMPILE_STATUS, &result);
			if(result != GL_TRUE)
			{
				char infoLog[GL_INFO_LOG_LENGTH] = {};
				GLsizei logLength = 0;
				glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &logLength);
				glGetShaderInfoLog(shaderID, logLength, 0, infoLog);
				const std::string errorMessage = format("Shader compile error: %s", infoLog);
				throw std::runtime_error(errorMessage);
			}
		}



		void CheckProgram(GLint programID)
		{
			GLint result = 0;
			glGetProgramiv(programID, GL_LINK_STATUS, &result);
			if(result != GL_TRUE)
			{
				char infoLog[GL_INFO_LOG_LENGTH] = {};
				GLsizei logLength = 0;
				glGetShaderiv(programID, GL_INFO_LOG_LENGTH, &logLength);
				glGetShaderInfoLog(programID, logLength, 0, infoLog);
				const std::string errorMessage = format("Shader link error: %s", infoLog);
				throw std::runtime_error(errorMessage);
			}
		}
	}



	Shader::Shader(const std::string& vertexFile, const std::string& fragmentFile) :
		mVertexFile(vertexFile),
		mFragmentFile(fragmentFile)
	{
		Compile();
	}



	Shader::~Shader()
	{
		glDetachShader(mShaderID, mVertexShaderID);
		glDetachShader(mShaderID, mFragmentShaderID);
		glDeleteShader(mVertexShaderID);
		glDeleteShader(mFragmentShaderID);
		glDeleteProgram(mShaderID);
	}



	void Shader::Compile()
	{
		// delete old shaders
		if(mVertexShaderID || mFragmentShaderID || mShaderID)
		{
			glDetachShader(mShaderID, mVertexShaderID);
			glDetachShader(mShaderID, mFragmentShaderID);
			glDeleteShader(mVertexShaderID);
			glDeleteShader(mFragmentShaderID);
			glDeleteProgram(mShaderID);
		}

		// load vertex shader
		const std::string vertCode = ReadFile(mVertexFile);
		if(vertCode.empty())
			throw std::runtime_error(format("File \"%s\" not found or empty.", mVertexFile.c_str()));

		// load fragment shader
		const std::string fragCode = ReadFile(mFragmentFile);
		if(fragCode.empty())
			throw std::runtime_error(format("File \"%s\" not found or empty.", mFragmentFile.c_str()));

		// compile vertex shader
		const char* vertCodeCStr = vertCode.c_str();
		GLint vertCodeSize = static_cast<GLint>(vertCode.size());
		mVertexShaderID = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(mVertexShaderID, 1, &vertCodeCStr, &vertCodeSize);
		glCompileShader(mVertexShaderID);
		GL_CHECK();
		CheckShader(mVertexShaderID);

		// compile fragment shader
		const char* fragCodeCStr = fragCode.c_str();
		GLint fragCodeSize = static_cast<GLint>(fragCode.size());
		mFragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(mFragmentShaderID, 1, &fragCodeCStr, &fragCodeSize);
		glCompileShader(mFragmentShaderID);
		GL_CHECK();
		CheckShader(mFragmentShaderID);

		// link shaders
		mShaderID = glCreateProgram();
		glAttachShader(mShaderID, mVertexShaderID);
		glAttachShader(mShaderID, mFragmentShaderID);
		glBindAttribLocation(mShaderID, 0, "pos");
		glLinkProgram(mShaderID);
		GL_CHECK();
		CheckProgram(mShaderID);
	}



	void Shader::Bind()
	{
		glUseProgram(mShaderID);
	}



	void Shader::Unbind()
	{
		glUseProgram(0);
	}



	void Shader::Set(const std::string& name, float v)
	{
		glUniform1f(glGetUniformLocation(mShaderID, name.c_str()), v);
	}



	void Shader::Set(uint32_t slot, const std::string& name, GLTexture* tex)
	{
		glActiveTexture(GL_TEXTURE0 + slot);
		tex->Bind();
		glUniform1i(glGetUniformLocation(mShaderID, name.c_str()), slot);
	}
}
