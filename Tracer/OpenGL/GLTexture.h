#pragma once

// Project
#include "Utility/LinearMath.h"

// C++
#include <stdint.h>
#include <vector>

namespace Tracer
{
	class GLTexture
	{
	public:
		enum class Types
		{
			Byte4,
			Float4,
			Half4
		};

		explicit GLTexture(const int2& resolution, Types type);
		virtual ~GLTexture();

		virtual void Bind() const;
		virtual void Unbind() const;

		void Upload(const std::vector<uint32_t>& pixels);
		void Upload(const std::vector<float4>& pixels);
		void Upload(const std::vector<half4>& pixels);

		void Download(std::vector<uint32_t>& pixels) const;
		void Download(std::vector<float4>& pixels) const;
		void Download(std::vector<half4>& pixels) const;

		inline uint32_t ID() const { return mId; }
		inline Types Type() const { return mType; }
		inline int2 Resolution() const { return mResolution; }

	private:
		uint32_t mId = 0;
		Types mType = Types::Byte4;
		int2 mResolution;
	};
};
