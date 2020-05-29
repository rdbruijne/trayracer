#include "GLTexture.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// C++
#include <assert.h>

namespace Tracer
{
	GLTexture::GLTexture(const int2& resolution, Types type) :
		mType(type),
		mResolution(resolution)
	{
		glGenTextures(1, &mId);

		glBindTexture(GL_TEXTURE_2D, mId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		switch(mType)
		{
		case Types::Byte4:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mResolution.x, mResolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			break;

		case Types::Float4:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, nullptr);
			break;
		}

		glBindTexture(GL_TEXTURE_2D, 0);
	}



	GLTexture::~GLTexture()
	{
		glDeleteTextures(1, &mId);
	}



	void GLTexture::Bind()
	{
		glBindTexture(GL_TEXTURE_2D, mId);
	}



	void GLTexture::Unbind()
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}



	void GLTexture::Upload(const std::vector<uint32_t>& pixels)
	{
		assert(mType == Types::Byte4);
		assert(pixels.size() == static_cast<size_t>(mResolution.x) * mResolution.y);
		Bind();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mResolution.x, mResolution.y, 0, GL_BGRA, GL_UNSIGNED_BYTE, pixels.data());
		Unbind();
	}



	void GLTexture::Upload(const std::vector<float4>& pixels)
	{
		assert(mType == Types::Float4);
		assert(pixels.size() == static_cast<size_t>(mResolution.x) * mResolution.y);
		Bind();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, pixels.data());
		Unbind();
	}
}
