#pragma once

// Project
#include "OpenGL/GLTexture.h"
#include "Utility/LinearMath.h"

// C++
#include <cstdint>
#include <vector>

namespace Tracer
{
	class GLFramebuffer
	{
	public:
		explicit GLFramebuffer(const int2& resolution);
		~GLFramebuffer();

		void Bind() const;
		void Unbind() const;

		inline uint32_t ID() const { return mId; }
		inline GLTexture* Texture() const { return mGlTexture; }
		inline int2 Resolution() const { return mGlTexture->Resolution(); }

	private:
		uint32_t mId = 0;
		uint32_t mDrawBuffer = 0;
		GLTexture* mGlTexture = nullptr;
	};
};
