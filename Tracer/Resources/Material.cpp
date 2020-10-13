#include "Material.h"

namespace Tracer
{
	//----------------------------------------------------------------------------------------------------------------------------
	// Material Property
	//----------------------------------------------------------------------------------------------------------------------------
	void Material::Property::Set(const float3& color)
	{
		assert(mColorEnabled);
		mColor = color;
		MarkDirty();
	}



	void Material::Property::Set(std::shared_ptr<Texture> tex)
	{
		assert(mTextureEnabled);
		mTexture = tex;
		MarkDirty();
	}



	void Material::Property::Build()
	{
		if(mTexture && mTexture->IsDirty())
		{
			mTexture->Build();
			mTexture->MarkClean();
		}

		mCudaProperty.r = __float2half(mColor.x);
		mCudaProperty.g = __float2half(mColor.y);
		mCudaProperty.b = __float2half(mColor.z);
		mCudaProperty.textureMap = mTexture ? mTexture->CudaObject() : 0;
		mCudaProperty.useColor = mColorEnabled ? 1 : 0;
		mCudaProperty.useTexture = mTextureEnabled ? 1 : 0;
	}



	//----------------------------------------------------------------------------------------------------------------------------
	// Material
	//----------------------------------------------------------------------------------------------------------------------------
	Material::Material(const std::string& name) :
		Resource(name)
	{
		// set defaults
		SetProperty(PropertyIds::Diffuse, Property(true, true));
		SetProperty(PropertyIds::Emissive, Property(true, false));
		SetProperty(PropertyIds::Normal, Property(false, true));
	}



	void Material::Set(PropertyIds id, const float3& color)
	{
		mProperties[static_cast<size_t>(id)].Set(color);
		MarkDirty();
	}



	void Material::Set(PropertyIds id, std::shared_ptr<Texture> tex)
	{
		const size_t szId = static_cast<size_t>(id);
		RemoveDependency(mProperties[szId].TextureMap());
		mProperties[szId].Set(tex);
		AddDependency(mProperties[szId].TextureMap());
		MarkDirty();
	}



	void Material::Build()
	{
		for(size_t i = 0; i < magic_enum::enum_count<Material::PropertyIds>(); i++)
		{
			if(mProperties[i].IsDirty())
			{
				mProperties[i].Build();
				mProperties[i].MarkClean();
			}

		}
		mCudaMaterial.diffuse = mProperties[static_cast<size_t>(PropertyIds::Diffuse)].CudaProperty();
		mCudaMaterial.emissive = mProperties[static_cast<size_t>(PropertyIds::Emissive)].CudaProperty();
		mCudaMaterial.normal = mProperties[static_cast<size_t>(PropertyIds::Normal)].CudaProperty();
	}



	std::string ToString(Material::PropertyIds id)
	{
		return std::string(magic_enum::enum_name(id));
	}



	void Material::SetProperty(PropertyIds id, const Property& prop)
	{
		const size_t szId = static_cast<size_t>(id);
		RemoveDependency(mProperties[szId].TextureMap());
		mProperties[static_cast<size_t>(id)] = prop;
		AddDependency(mProperties[szId].TextureMap());
		MarkDirty();
	}
}
