#pragma once

// C++
#include <string>

namespace Tracer
{
	class GLTexture;
	class Shader
	{
	public:
		explicit Shader(const std::string& vertexFile, const std::string& fragmentFile);
		~Shader();

		void Compile();

		void Bind();
		void Unbind();

		void Set(const std::string& name, float v);
		void Set(const std::string& name, int v);
		void Set(uint32_t slot, const std::string& name, GLTexture* tex);

	private:
		std::string mVertexFile = "";
		std::string mFragmentFile = "";

		uint32_t mVertexShaderID = 0;
		uint32_t mFragmentShaderID = 0;
		uint32_t mShaderID = 0;
	};
}
