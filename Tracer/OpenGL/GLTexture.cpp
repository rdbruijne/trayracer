#include "GLTexture.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"


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
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mResolution.x, mResolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			break;

		case Types::Float4:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, 0);
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
}
