#include "Material.h"

namespace Tracer
{
	Material::Material(const std::string& name) :
		Resource(name)
	{
	}



	void Material::SetDiffuse(const float3& val)
	{
		mDiffuse = val;
		MarkDirty();
	}



	void Material::SetEmissive(const float3& val)
	{
		mEmissive = val;
		mEmissiveChanged = true;
		MarkDirty();
	}



	void Material::SetDiffuseMap(std::shared_ptr<Texture> tex)
	{
		RemoveDependency(mDiffuseMap);
		mDiffuseMap = tex->IsValid() ? tex : nullptr;
		AddDependency(mDiffuseMap);
		MarkDirty();
	}



	void Material::SetNormalMap(std::shared_ptr<Texture> tex)
	{
		RemoveDependency(mNormalMap);
		mNormalMap = tex->IsValid() ? tex : nullptr;
		AddDependency(mNormalMap);
		MarkDirty();
	}
}
