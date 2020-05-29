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
		MarkDirty();
		mEmissiveChanged = true;
	}



	void Material::SetDiffuseMap(std::shared_ptr<Texture> tex)
	{
		mDiffuseMap = tex;
		MarkDirty();
		AddDependency(tex);
	}
}
