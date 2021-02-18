#include "Material.h"

namespace Tracer
{
	//----------------------------------------------------------------------------------------------------------------------------
	// Material Property
	//----------------------------------------------------------------------------------------------------------------------------
	Material::Property& Material::Property::operator =(const Property& p)
	{
		mFlags = p.mFlags;
		mColor = p.mColor;
		mTexture = p.mTexture;
		MarkDirty();
		return *this;
	}



	void Material::Property::Set(float color)
	{
		assert(IsFloatColorEnabled());
		mColor.x = color;
		MarkDirty();
	}



	void Material::Property::Set(const float3& color)
	{
		assert(IsRgbColorEnabled());
		mColor = color;
		MarkDirty();
	}



	void Material::Property::Set(std::shared_ptr<Texture> tex)
	{
		assert(IsTextureEnabled());
		mTexture = tex;
		MarkDirty();
	}



	void Material::Property::Build()
	{
		std::lock_guard<std::mutex> l(mBuildMutex);

		// dirty check
		if(!IsDirty())
			return;

		// build texture
		if(mTexture && mTexture->IsDirty())
			mTexture->Build();

		// assign data
		mCudaProperty.r = __float2half(mColor.x);
		mCudaProperty.g = __float2half(mColor.y);
		mCudaProperty.b = __float2half(mColor.z);
		mCudaProperty.textureMap = mTexture ? mTexture->CudaObject() : 0;
		mCudaProperty.useColor   = (IsFloatColorEnabled() || IsRgbColorEnabled()) ? 1 : 0;
		mCudaProperty.useTexture = IsTextureEnabled() ? 1 : 0;

		// mark clean
		MarkClean();
	}



	//----------------------------------------------------------------------------------------------------------------------------
	// Material
	//----------------------------------------------------------------------------------------------------------------------------
	Material::Material(const std::string& name) :
		Resource(name)
	{
		// set defaults
		SetProperty(MaterialPropertyIds::Anisotropic,     Property(0, true));
		SetProperty(MaterialPropertyIds::Clearcoat,       Property(0, true));
		SetProperty(MaterialPropertyIds::ClearcoatGloss,  Property(0, true));
		SetProperty(MaterialPropertyIds::Diffuse,         Property(make_float3(1), true));
		SetProperty(MaterialPropertyIds::Emissive,        Property(make_float3(0), false));
		SetProperty(MaterialPropertyIds::Metallic,        Property(0, true));
		SetProperty(MaterialPropertyIds::Normal,          Property(Property::Flags::TexMap));
		SetProperty(MaterialPropertyIds::Roughness,       Property(1, true));
		SetProperty(MaterialPropertyIds::Sheen,           Property(0, true));
		SetProperty(MaterialPropertyIds::SheenTint,       Property(0, true));
		SetProperty(MaterialPropertyIds::Specular,        Property(0, true));
		SetProperty(MaterialPropertyIds::SpecularTint,    Property(0, true));
		SetProperty(MaterialPropertyIds::Subsurface,      Property(0, true));
	}



	void Material::Set(MaterialPropertyIds id, float color)
	{
		mProperties[static_cast<size_t>(id)].Set(color);
		MarkDirty();
	}



	void Material::Set(MaterialPropertyIds id, const float3& color)
	{
		mProperties[static_cast<size_t>(id)].Set(color);
		MarkDirty();
	}



	void Material::Set(MaterialPropertyIds id, std::shared_ptr<Texture> tex)
	{
		const size_t szId = static_cast<size_t>(id);
		RemoveDependency(mProperties[szId].TextureMap());
		mProperties[szId].Set(tex);
		AddDependency(mProperties[szId].TextureMap());
		MarkDirty();
	}



	void Material::Build()
	{
		std::lock_guard<std::mutex> l(mBuildMutex);

		// dirty check
		if(!IsDirty())
			return;

		// build properties
		for(size_t i = 0; i < static_cast<size_t>(MaterialPropertyIds::_Count); i++)
			if(mProperties[i].IsDirty())
				mProperties[i].Build();

		// assign properties
		for(size_t i = 0; i < static_cast<size_t>(MaterialPropertyIds::_Count); i++)
			mCudaMaterial.properties[i] = mProperties[i].CudaProperty();

		// mark clean
		MarkClean();
	}



	void Material::SetProperty(MaterialPropertyIds id, const Property& prop)
	{
		const size_t szId = static_cast<size_t>(id);
		RemoveDependency(mProperties[szId].TextureMap());
		mProperties[static_cast<size_t>(id)] = prop;
		AddDependency(mProperties[szId].TextureMap());
		MarkDirty();
	}
}
