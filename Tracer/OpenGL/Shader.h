#pragma once

#include "Utility/Named.h"

// C++
#include <map>
#include <string>
#include <vector>

namespace Tracer
{
	class GLTexture;
	//--------------------------------------------------------------------------------------------------------------------------
	// Shader
	//--------------------------------------------------------------------------------------------------------------------------
	class Shader : public Named
	{
	public:
		enum class SourceType
		{
			Code,
			File
		};

		class Uniform;

		explicit Shader(const std::string& name, const std::string& vertex, SourceType vertexSourceType, const std::string& fragment, SourceType fragmentSourceType);
		~Shader();

		void Compile();

		void Bind();
		void Unbind();

		void ApplyUniforms();

		// check for compile/link errors
		bool IsValid() const { return mIsValid; }
		const std::string& ErrorLog() const { return mErrorLog; }

		// enable/disable shader
		bool IsEnabled() { return mEnabled; }
		void SetEnabled(bool enabled) { mEnabled = enabled; }

		// vertex part
		SourceType VertexSourceType() const { return mVertexSourceType; }
		const std::string& VertexFile() const { return mVertexFile; }
		const std::string& VertexCode() const { return mVertexCode; }

		// fragment part
		SourceType FragmentSourceType() const { return mFragmentSourceType; }
		const std::string& FragmentFile() const { return mFragmentFile; }
		const std::string& FragmentCode() const { return mFragmentCode; }

		// deal with uniforms
		std::map<std::string, Uniform>& Uniforms() { return mUniforms; }
		const std::map<std::string, Uniform>& Uniforms() const { return mUniforms; }

		Uniform* GetUniform(const std::string name);
		const Uniform* GetUniform(const std::string name) const;

		template<typename TYPE>
		void Set(const std::string& name, const TYPE& v);
		void Set(const std::string& name, uint32_t slot, GLTexture* tex);

		// check if uniform is internal to engine
		static bool IsInternalUniform(const std::string& name);

		// built-in shader code
		static const std::string& FullScreenQuadFrag();
		static const std::string& FullScreenQuadVert();

	private:
		void Unload();
		void ParseUniforms();
		void ParseDescriptor();

		bool mEnabled = true;
		bool mIsValid = false;

		SourceType mVertexSourceType;
		std::string mVertexFile = "";
		std::string mVertexCode = "";

		SourceType mFragmentSourceType;
		std::string mFragmentFile = "";
		std::string mFragmentCode = "";

		uint32_t mVertexShaderID = 0;
		uint32_t mFragmentShaderID = 0;
		uint32_t mShaderID = 0;

		std::string mErrorLog = "";
		std::map<std::string, Uniform> mUniforms;
	};



	//--------------------------------------------------------------------------------------------------------------------------
	// Shader uniforms
	//--------------------------------------------------------------------------------------------------------------------------
	class Shader::Uniform
	{
		friend class Shader;
	public:
		enum class Types
		{
			Unknown,
			Float,
			Int,
			Texture
		};

		Uniform() = default;
		explicit Uniform(Types type) : mType(type) {}

		Types Type() const { return mType; }

		// enum type
		bool IsEnum() const { return mType == Types::Int && mEnumKeys.size() > 0; }
		const std::vector<std::string>& EnumKeys() const { return mEnumKeys; }
		void SetEnumKeys(const std::vector<std::string>& keys) { mEnumKeys = keys; }

		// range info
		bool HasRange() const { return mHasRange; }

		// log scale slider?
		bool IsLogarithmic() const { return mIsLogarithmic; }
		void SetLogarithmic(bool logarithmic) { mIsLogarithmic = logarithmic; }

		// float
		void Get(float* v) const;
		void Set(float v);

		void GetRange(float* min, float* max) const;
		void SetRange(float min, float max);

		// int
		void Get(int* v) const;
		void Set(int v);

		void GetRange(int* min, int* max) const;
		void SetRange(int min, int max);

		// texture
		void Get(uint32_t* slot, GLTexture** tex) const;
		void Set(uint32_t slot, GLTexture* tex);

	private:
		Types mType = Types::Unknown;

		// value
		union Data
		{
			float f;
			int i;
			struct Tex
			{
				uint32_t slot;
				GLTexture* t;
			} t;
		} mData = {};

		// value range
		bool mHasRange = false;
		union Range
		{
			int i[2];
			float f[2];
		} mRange;

		// log scale
		bool mIsLogarithmic = false;

		// enum
		std::vector<std::string> mEnumKeys;
	};



	//--------------------------------------------------------------------------------------------------------------------------
	// Shader inlines
	//--------------------------------------------------------------------------------------------------------------------------
	template<typename TYPE>
	void Shader::Set(const std::string& name, const TYPE& v)
	{
		mUniforms[name].Set(v);
	}



	inline void Shader::Set(const std::string& name, uint32_t slot, GLTexture* tex)
	{
		mUniforms[name].Set(slot, tex);
	}
}
