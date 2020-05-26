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

		// values
		MATERIAL_VALUE(float3, Diffuse, make_float3(.5f));
		MATERIAL_VALUE(float3, Emissive, make_float3(0));
		//MATERIAL_VALUE(float,  Opacity, 1);
		//MATERIAL_VALUE(float,  RefractI, 1);
		//MATERIAL_VALUE(float,  Shininess, 1);
		//MATERIAL_VALUE(float3, Specular, make_float3(0));
		//MATERIAL_VALUE(float3, Transparent, make_float3(1));

		// textures
		///MATERIAL_TEXTURE(BaseColorMap);
		MATERIAL_TEXTURE(DiffuseMap);
		//MATERIAL_TEXTURE(DiffuseRoughnessMap);
		//MATERIAL_TEXTURE(DisplacementMap);
		//MATERIAL_TEXTURE(EmissionColorMap);
		//MATERIAL_TEXTURE(EmissiveMap);
		//MATERIAL_TEXTURE(HeightMap);
		//MATERIAL_TEXTURE(MetalnessMap);
		//MATERIAL_TEXTURE(NormalCameraMap);
		//MATERIAL_TEXTURE(NormalMap);
		//MATERIAL_TEXTURE(OpacityMap);
		//MATERIAL_TEXTURE(ShininessMap);
		//MATERIAL_TEXTURE(SpecularMap);
	};
}

#undef MATERIAL_VALUE
#undef MATERIAL_TEXTURE
