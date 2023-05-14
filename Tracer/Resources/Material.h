#pragma once

// Project
#include "Common/CommonStructs.h"
#include "Resources/Resource.h"
#include "Resources/Texture.h"
#include "Utility/Enum.h"
#include "Utility/LinearMath.h"

// C++
#include <memory>
#include <mutex>
#include <string>

namespace Tracer
{
	class Renderer;
	class Texture;
	class Material : public Resource
	{
		// material property
		class Property : public Defilable
		{
		public:
			enum class Flags
			{
				None		= 0x0,
				ColorFloat	= 0x1,
				ColorRgb	= 0x2,
				TexMap		= 0x4
			};

			Property() = default;
			Property(const Property& p);
			explicit inline Property(Flags flags);
			explicit inline Property(float c, bool textureEnabled, float minVal = 0, float maxVal = 1);
			explicit inline Property(const float3& c, bool textureEnabled);

			Property& operator =(const Property& p);

			// float color
			inline bool IsFloatColorEnabled() const;
			inline float FloatColor() const;
			inline float2 FloatColorRange() const;
			void Set(float color);

			// rgb color
			inline bool IsRgbColorEnabled() const;
			inline const float3& RgbColor() const;
			void Set(const float3& color);

			// texture map
			inline bool IsTextureEnabled() const;
			inline std::shared_ptr<Texture> TextureMap() const;
			void Set(std::shared_ptr<Texture> tex);

			// build
			void Build();
			void Upload(Renderer* renderer);

			inline const CudaMaterialProperty& CudaProperty() const;

		private:
			Flags mFlags = Flags::None;
			float3 mColor = make_float3(0.f);
			std::shared_ptr<Texture> mTexture = nullptr;

			// mutex
			std::mutex mMutex;

			// build data
			CudaMaterialProperty mCudaProperty = {};
		};

		// disable copying
		Material(const Material&) = delete;
		Material& operator =(const Material&) = delete;

	public:
		// construction
		explicit Material(const std::string& name);

		// float color
		inline bool IsFloatColorEnabled(MaterialPropertyIds id) const;
		inline float FloatColor(MaterialPropertyIds id) const;
		inline float2 FloatColorRange(MaterialPropertyIds id) const;
		void Set(MaterialPropertyIds id, float color);

		// rgb color
		inline bool IsRgbColorEnabled(MaterialPropertyIds id) const;
		inline const float3& RgbColor(MaterialPropertyIds id) const;
		void Set(MaterialPropertyIds id, const float3& color);

		// texture
		inline bool IsTextureEnabled(MaterialPropertyIds id) const;
		inline std::shared_ptr<Texture> TextureMap(MaterialPropertyIds id) const;
		void Set(MaterialPropertyIds id, std::shared_ptr<Texture> tex);

		// info
		inline bool EmissiveChanged() const;

		// build
		void Build();

		// upload
		void Upload(Renderer* renderer);

		// build info
		inline const CudaMatarial& CudaMaterial() const;

	private:
		inline const Property& GetProperty(MaterialPropertyIds id) const;
		void SetProperty(MaterialPropertyIds id, const Property& prop);

		std::array<Property, magic_enum::enum_count<MaterialPropertyIds>()> mProperties = {};

		// mutex
		std::mutex mMutex;

		// GPU data
		CudaMatarial mCudaMaterial = {};
	};

	ENUM_BITWISE_OPERATORS(Material::Property::Flags);
}

#include "Material.inl"
