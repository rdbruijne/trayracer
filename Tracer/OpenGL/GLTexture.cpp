#include "GLTexture.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// C++
#include <cassert>

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

		case Types::Half4:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, nullptr);
			break;
		}

		glBindTexture(GL_TEXTURE_2D, 0);
	}



	GLTexture::~GLTexture()
	{
		glDeleteTextures(1, &mId);
	}



	void GLTexture::Bind() const
	{
		glBindTexture(GL_TEXTURE_2D, mId);
	}



	void GLTexture::Unbind() const
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}



	void GLTexture::BindEmpty()
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



	void GLTexture::Upload(const std::vector<half4>& pixels)
	{
		assert(mType == Types::Half4);
		assert(pixels.size() == static_cast<size_t>(mResolution.x) * mResolution.y);
		Bind();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_HALF_FLOAT, pixels.data());
		Unbind();
	}



	void GLTexture::Download(std::vector<uint32_t>& pixels) const
	{
		pixels.resize(static_cast<size_t>(mResolution.x) * mResolution.y);

		Bind();
		glReadPixels(0, 0, mResolution.x, mResolution.y, GL_BGRA, GL_UNSIGNED_BYTE, pixels.data());
		Unbind();
	}



	void GLTexture::Download(std::vector<float4>& pixels) const
	{
		pixels.resize(static_cast<size_t>(mResolution.x) * mResolution.y);

		Bind();
		glReadPixels(0, 0, mResolution.x, mResolution.y, GL_RGBA, GL_FLOAT, pixels.data());
		Unbind();
	}



	void GLTexture::Download(std::vector<half4>& pixels) const
	{
		pixels.resize(static_cast<size_t>(mResolution.x) * mResolution.y);

		Bind();
		glReadPixels(0, 0, mResolution.x, mResolution.y, GL_RGBA, GL_HALF_FLOAT, pixels.data());
		Unbind();
	}
}
