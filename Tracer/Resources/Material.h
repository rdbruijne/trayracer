#pragma once

// Project
#include "Resources/Resource.h"
#include "Resources/Texture.h"
#include "Utility/LinearMath.h"

// C++
#include <memory>
#include <string>



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
		const float3& Diffuse() const { return mDiffuse; }
		const float3& Emissive() const { return mEmissive; }

		const std::shared_ptr<Texture>& DiffuseMap() const { return mDiffuseMap; }
		const std::shared_ptr<Texture>& NormalMap() const { return mNormalMap; }

		//----------------
		// setters
		//----------------
		void SetDiffuse(const float3& val);
		void SetEmissive(const float3& val);

		void SetDiffuseMap(std::shared_ptr<Texture> tex);
		void SetNormalMap(std::shared_ptr<Texture> tex);

	private:
		bool mEmissiveChanged = false;

		float3 mDiffuse = make_float3(.5f);
		float3 mEmissive = make_float3(0);

		std::shared_ptr<Texture> mDiffuseMap = nullptr;
		std::shared_ptr<Texture> mNormalMap = nullptr;
	};
}
