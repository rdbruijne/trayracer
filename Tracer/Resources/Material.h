#pragma once

// Project
#include "Common/CommonStructs.h"
#include "Resources/Resource.h"
#include "Resources/Texture.h"
#include "Utility/LinearMath.h"

// C++
#include <memory>
#include <mutex>
#include <string>

namespace Tracer
{
	class Texture;
	class Material : public Resource
	{
		class Property : public Defilable
		{
		public:
			Property() = default;
			explicit Property(bool colorEnabled, bool textureEnabled) : mColorEnabled(colorEnabled), mTextureEnabled(textureEnabled) {}
			explicit Property(const float3& c) : mColorEnabled(true), mColor(c) {}
			explicit Property(std::shared_ptr<Texture> t) : mTextureEnabled(true), mTexture(t) {}
			explicit Property(const float3& c, std::shared_ptr<Texture> t) : mColorEnabled(true), mColor(c), mTextureEnabled(true), mTexture(t) {}

			Property& operator =(const Property& p);

			// color
			inline bool IsColorEnabled() const { return mColorEnabled; }
			inline const float3& Color() const { return mColor; }
			void Set(const float3& color);

			// texture map
			inline bool IsTextureEnabled() const { return mTextureEnabled; }
			inline std::shared_ptr<Texture> TextureMap() const { return mTexture; }
			void Set(std::shared_ptr<Texture> tex);

			// build
			void Build();
			const CudaMaterialProperty& CudaProperty() const { return mCudaProperty; }

		private:
			// color
			bool mColorEnabled = false;
			float3 mColor = make_float3(0.f);

			// texture map
			bool mTextureEnabled = false;
			std::shared_ptr<Texture> mTexture = nullptr;

			// build data
			std::mutex mBuildMutex;
			CudaMaterialProperty mCudaProperty;
		};

	public:
		// construction
		explicit Material(const std::string& name);

		Material& operator =(const Material& t) = delete;

		// properties
		enum class PropertyIds
		{
			Diffuse,
			Emissive,
			Normal
		};

		inline bool IsColorEnabled(PropertyIds id) { return mProperties[static_cast<size_t>(id)].IsColorEnabled(); }
		inline const float3& GetColor(PropertyIds id) const { return mProperties[static_cast<size_t>(id)].Color(); }
		void Set(PropertyIds id, const float3& color);

		inline bool IsTextureEnabled(PropertyIds id) { return mProperties[static_cast<size_t>(id)].IsTextureEnabled(); }
		inline std::shared_ptr<Texture> GetTextureMap(PropertyIds id) const { return mProperties[static_cast<size_t>(id)].TextureMap(); }
		void Set(PropertyIds id, std::shared_ptr<Texture> tex);

		// info
		bool EmissiveChanged() const { return mProperties[static_cast<size_t>(PropertyIds::Emissive)].IsDirty(); }

		// build
		void Build();

		// build info
		inline const CudaMatarial& CudaMaterial() const { return mCudaMaterial; }

	private:
		const Property& GetProperty(PropertyIds id) const { return mProperties[static_cast<size_t>(id)]; }
		void SetProperty(PropertyIds id, const Property& prop);

		std::array<Property, magic_enum::enum_count<Material::PropertyIds>()> mProperties = {};

		// build data
		std::mutex mBuildMutex;
		CudaMatarial mCudaMaterial = {};
	};

	std::string ToString(Material::PropertyIds id);
}
