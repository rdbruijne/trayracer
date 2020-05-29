#pragma once

// Project
#include "Resources/Resource.h"
#include "Resources/Texture.h"
#include "Utility/LinearMath.h"

// C++
#include <memory>
#include <string>

// material texture property
#define MATERIAL_TEXTURE(name)																			\
	public: const std::shared_ptr<Texture>& name() const { return m##name; }							\
	public: void Set##name(std::shared_ptr<Texture> t) { m##name = t;  MarkDirty(); AddDependency(t); }	\
	private: std::shared_ptr<Texture> m##name = nullptr;



// material value property
#define MATERIAL_VALUE(type, name, defaultValue)						\
	public: const type& name() const { return m##name; }				\
	public: void Set##name(const type& t) { m##name = t; MarkDirty(); }	\
	private: type m##name = defaultValue;



namespace Tracer
{
	class Texture;
	class Material : public Resource
	{
	public:
		explicit Material(const std::string& name);

		//----------------
		// info
		//----------------
		bool EmissiveChanged() const { return mEmissiveChanged; }
		void ResetEmissiveChanged() { mEmissiveChanged = false; }


		//----------------
		// getters
		//----------------
		const float3 Diffuse() const { return mDiffuse; }
		const float3 Emissive() const { return mEmissive; }

		const std::shared_ptr<Texture>& DiffuseMap() const { return mDiffuseMap; }

		//----------------
		// setters
		//----------------
		void SetDiffuse(const float3& val);
		void SetEmissive(const float3& val);

		void SetDiffuseMap(std::shared_ptr<Texture> tex);

	private:
		bool mEmissiveChanged = false;

		float3 mDiffuse = make_float3(.5f);
		float3 mEmissive = make_float3(0);

		std::shared_ptr<Texture> mDiffuseMap = nullptr;
	};
}

#undef MATERIAL_VALUE
#undef MATERIAL_TEXTURE
