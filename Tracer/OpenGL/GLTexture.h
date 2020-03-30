#pragma once

// Project
#include "Utility/LinearMath.h"

// C++
#include <stdint.h>

namespace Tracer
{
	class GLTexture
	{
	public:
		enum class Types
		{
			Byte4,
			Float4
		};

		explicit GLTexture(const int2& resolution, Types type);
		~GLTexture();

		void Bind();
		void Unbind();

		inline uint32_t GetID() const { return mId; }
		inline Types GetType() const { return mType; }
		inline int2 GetResolution() const { return mResolution; }

	private:
		uint32_t mId = 0;
		Types mType = Types::Byte4;
		int2 mResolution;
	};
};
