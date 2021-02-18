namespace Tracer
{
	//--------------------------------------------------------------------------------------------------------------------------
	// Material::Property
	//--------------------------------------------------------------------------------------------------------------------------
	Material::Property::Property(Flags flags) :
		mFlags(flags)
	{
	}



	Material::Property::Property(float c, bool textureEnabled, float minVal, float maxVal) :
		mFlags(Flags::ColorFloat),
		mColor(make_float3(c, minVal, maxVal))
	{
		if(textureEnabled)
			mFlags = mFlags | Flags::TexMap;
	}



	Material::Property::Property(const float3& c, bool textureEnabled) :
		mFlags(Flags::ColorRgb),
		mColor(c)
	{
		if(textureEnabled)
			mFlags = mFlags | Flags::TexMap;
	}



	bool Material::Property::IsFloatColorEnabled() const
	{
		return (mFlags & Flags::ColorFloat) == Flags::ColorFloat;
	}



	float Material::Property::FloatColor() const
	{
		return mColor.x;
	}



	float2 Material::Property::FloatColorRange() const
	{
		return make_float2(mColor.y, mColor.z);
	}



	bool Material::Property::IsRgbColorEnabled() const
	{
		return (mFlags & Flags::ColorRgb) == Flags::ColorRgb;
	}



	const float3& Material::Property::RgbColor() const
	{
		return mColor;
	}



	bool Material::Property::IsTextureEnabled() const
	{
		return (mFlags & Flags::TexMap) == Flags::TexMap;
	}



	std::shared_ptr<Texture> Material::Property::TextureMap() const
	{
		return mTexture;
	}



	const CudaMaterialProperty& Material::Property::CudaProperty() const
	{
		return mCudaProperty;
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// Material
	//--------------------------------------------------------------------------------------------------------------------------
	bool Material::IsFloatColorEnabled(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)].IsFloatColorEnabled();
	}



	float Material::FloatColor(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)].FloatColor();
	}



	float2 Material::FloatColorRange(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)].FloatColorRange();
	}



	bool Material::IsRgbColorEnabled(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)].IsRgbColorEnabled();
	}



	const float3& Material::RgbColor(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)].RgbColor();
	}



	bool Material::IsTextureEnabled(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)].IsTextureEnabled();
	}



	std::shared_ptr<Texture> Material::TextureMap(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)].TextureMap();
	}



	bool Material::EmissiveChanged() const
	{
		return mProperties[static_cast<size_t>(MaterialPropertyIds::Emissive)].IsDirty();
	}



	const CudaMatarial& Material::CudaMaterial() const
	{
		return mCudaMaterial;
	}



	const Material::Property& Material::GetProperty(MaterialPropertyIds id) const
	{
		return mProperties[static_cast<size_t>(id)];
	}
}
