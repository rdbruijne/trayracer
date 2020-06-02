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
		mDiffuseMap = tex;
		AddDependency(tex);
		MarkDirty();
	}



	void Material::SetNormalMap(std::shared_ptr<Texture> tex)
	{
		mNormalMap = tex;
		AddDependency(tex);
		MarkDirty();
	}
}
